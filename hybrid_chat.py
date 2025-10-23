# hybrid_chat.py
import os
import json
from typing import List
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
import config
import asyncio
import textwrap
from rich.console import Console
from rich.progress import Progress
from rich.panel import Panel
import re

console = Console()

# Keep in-session chat history
history = []
from neo4j import AsyncGraphDatabase  # <-- use AsyncGraphDatabase
# -----------------------------
# Config
# -----------------------------
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5
INDEX_NAME = config.PINECONE_INDEX_NAME

# -----------------------------
# Initialize clients
# -----------------------------
client = OpenAI(api_key=config.OPENAI_API_KEY)
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# Connect to Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating managed index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=config.PINECONE_VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east1-gcp")
    )

index = pc.Index(INDEX_NAME)

# Connect to Neo4j
driver = AsyncGraphDatabase.driver(
    config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)
# -----------------------------
# Helper functions
# -----------------------------


CACHE_FILE = "embedding_cache.json"

# Load cache if exists
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        embedding_cache = json.load(f)
else:
    embedding_cache = {}

def save_cache():
    """Save embedding cache to disk."""
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(embedding_cache, f)

# -----------------------------
# Async embedding
# -----------------------------
async def embed_text_async(text: str) -> List[float]:
    """Get embedding with caching (async compatible)."""
    text_key = text.strip().lower()  # normalize
    if text_key in embedding_cache:
        console.print(f"[cyan]Using cached embedding for:[/cyan] {text_key}")
        return embedding_cache[text_key]

    # Compute embedding in executor
    loop = asyncio.get_event_loop()
    try:
        resp = await loop.run_in_executor(
            None,
            lambda: client.embeddings.create(model=EMBED_MODEL, input=[text])
        )
        emb = resp.data[0].embedding
        embedding_cache[text_key] = emb
        save_cache()
        console.print(f"[green]Cached new embedding for:[/green] {text_key}")
        return emb
    except Exception as e:
        console.print(f"[red]Error embedding text:[/red] {e}")
        return None

# -----------------------------
# Async Pinecone query
# -----------------------------
async def pinecone_query_async(query_text: str, top_k=TOP_K):
    vec = await embed_text_async(query_text)
    # Pinecone API is still sync; run in executor
    loop = asyncio.get_event_loop()
    res = await loop.run_in_executor(None, lambda: index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    ))
    print("DEBUG: Pinecone top 5 results:", len(res["matches"]))
    return res["matches"]

# -----------------------------
# Async Neo4j fetch
# -----------------------------
async def fetch_graph_context_async(node_ids: List[str], neighborhood_depth=1):
    """Fetch neighboring nodes from Neo4j asynchronously."""
    facts = []

    async with driver.session() as session:
        for nid in node_ids:
            # Fetch the source node's name
            q_name = "MATCH (n:Entity {id:$nid}) RETURN n.name AS name LIMIT 1"
            rec = await session.run(q_name, nid=nid)
            src = await rec.single()
            source_name = src["name"] if src else nid

            # Fetch relationships
            q = (
                "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
                "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, "
                "m.name AS name, m.type AS type, m.description AS description "
                "LIMIT 10"
            )
            recs = await session.run(q, nid=nid)
            async for r in recs:
                facts.append({
                    "source": nid,
                    "source_name": source_name,
                    "rel": r["rel"],
                    "target_id": r["id"],
                    "target_name": r["name"],
                    "target_desc": (r["description"] or "")[:400],
                    "labels": r["labels"]
                })

    print("DEBUG: Graph facts:", len(facts))
    return facts


def clean_itinerary(raw_text: str) -> str:
    """
    Format raw itinerary text into proper Markdown with headings and bullets.
    """
    text = raw_text

    # Step 1: Replace "####" with "##" (subheadings)
    text = re.sub(r'####\s*', '## ', text)

    # Step 2: Replace "###" with "#" (main heading)
    text = re.sub(r'###\s*', '# ', text)

    # Step 3: Split Morning/Afternoon/Evening into bullet points
    text = re.sub(r'\*\*Morning\*\*:\s*', '**Morning:**\n- ', text)
    text = re.sub(r'\*\*Afternoon\*\*:\s*', '**Afternoon:**\n- ', text)
    text = re.sub(r'\*\*Evening\*\*:\s*', '**Evening:**\n- ', text)

    # Step 4: Replace hyphen-separated sentences after time blocks with bullets
    # (optional: add line breaks after periods)
    text = re.sub(r'\s*-\s+', '\n- ', text)

    # Step 5: Ensure proper line breaks for each paragraph
    text = re.sub(r'\s{2,}', ' ', text)  # remove multiple spaces
    text = re.sub(r'(\.)(\s*)([A-Z])', r'\1\n\3', text)  # break after periods before capital letters

    return text.strip()


def build_prompt(user_query, pinecone_matches, graph_facts):
    """
    Build a chat prompt combining semantic search results and graph facts,
    with chain-of-thought style reasoning instructions for better context understanding.
    """
    system = (
        "You are a professional AI travel planner and reasoning assistant. "
        "You have access to structured travel data from a vector database (semantic matches) "
        "and a graph database (relations between cities, attractions, and activities). "
        "Use reasoning to infer the most relevant itinerary or information for the user's query. "
        "Do NOT reveal your reasoning steps explicitly in the final answer."
        " Provide clear, concise, and well-structured responses."
        "Do not use attractions ids in the final answer."
    )

    # --- Build semantic context from Pinecone ---
    vec_context = []
    for m in pinecone_matches:
        meta = m["metadata"]
        snippet = f"- {meta.get('name', '')} ({meta.get('type', '')}), city: {meta.get('city', '')}. {meta.get('description', '')[:150]}"
        vec_context.append(snippet)

    # --- Build graph context from Neo4j ---
    graph_context = [
        f"- {f['source_name']} → ({f['rel']}) → {f['target_name']}: {f['target_desc']}"
        for f in graph_facts
    ]

    # --- Chain-of-Thought Reasoning Instructions ---
    reasoning_instructions = (
        "First, silently reason through the provided data:\n"
        "1. Identify the user's intent (e.g., trip planning, attraction comparison, recommendations).\n"
        "2. Select the most relevant cities or attractions based on their type, description, and tags.\n"
        "3. Create a logical itinerary or direct answer connecting related attractions or cities.\n"
        "4. Finally, summarize your reasoning into a clear and well-structured response for the user.\n"
        "Do not show reasoning steps explicitly — only the final, polished result."
    )

    # --- Construct the prompt ---
    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content":
         f"User Query: {user_query}\n\n"
         f"{reasoning_instructions}\n\n"
         "Semantic Context (from Vector DB):\n" + "\n".join(vec_context[:10]) + "\n\n"
         "Graph Context (from Neo4j):\n" + "\n".join(graph_context[:20]) + "\n\n"
         "Based on the context above, provide a clear and concise final answer to the user's question.\n"
         "The answer should be well-structured and separated sections."
         "=== Reasoning ===\n"
        "Briefly explain your reasoning, assumptions, and why this approach is optimal.\n"
         "If applicable,include 2–3 practical suggestions, tips, or actionable steps to help the user implement or experience your recommendation."
        }
    ]
    return prompt

def call_chat(prompt_messages):
    """Call OpenAI and return both reasoning and answer separately."""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=prompt_messages,
        max_tokens=750,
        temperature=0.1
    )
    answer = resp.choices[0].message.content
    return answer
# -----------------------------
# Async interactive chat
# -----------------------------
import asyncio
import textwrap
from rich.console import Console
from rich.progress import Progress
from rich.panel import Panel

console = Console()

# Keep in-session chat history
history = []
from rich.console import Console
from rich.progress import Progress

console = Console()
history = []
async def interactive_chat_async():
    console.print("🛫 [bold cyan]Hybrid Travel Assistant[/bold cyan]. Type 'exit' or 'quit' to leave.", style="bold")
   
    while True:
        query = input("\n Enter your travel question:").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            console.print("\n[bold green]Exiting chat. Safe travels![/bold green]")
            break
        with Progress() as progress: 
            pinecone_task = progress.add_task("[cyan]Fetching Pinecone matches...", total=1)
            graph_task = progress.add_task("[cyan]Fetching graph context...", total=1)
            # Step 2: Fetch Pinecone matches (async)
            try:
                matches = await pinecone_query_async(query)
                progress.update(pinecone_task, completed=1)
            except Exception as e:
                console.print(f"[red]Error fetching Pinecone matches: {e}[/red]")
                continue
                # Step 3: Fetch Graph context (async)
            try:
                match_ids = [m["id"] for m in matches]
                graph_facts = await fetch_graph_context_async(match_ids)
                progress.update(graph_task, completed=1)
            except Exception as e:
                console.print(f"[red]Error fetching graph context: {e}[/red]")
                continue
        prompt = build_prompt(query, matches, graph_facts)
        #     # Call the chat model
        answer = call_chat(prompt)
        # answer = clean_itinerary(answer)
        # Store the Q&A in session history
        history.append({"user": query, "assistant": answer})
        print(answer)
    #     console.print(
    # Panel(
    #     textwrap.fill(answer, width=console.width - 5),
    #     title="[bold green]Assistant Answer[/bold green]",
    #     border_style="green",
    #     width=console.width
    # ))

        
if __name__ == "__main__":
    asyncio.run(interactive_chat_async())
