"""
MCP Server Concepts Tutorial — Flask Web App
Architecture follows SOLID principles:
  S: Each class has one responsibility (data, evaluation, rendering)
  O: New concepts/questions added via data, not code changes
  L: QuizQuestion subtypes are fully substitutable
  I: Renderer only knows about what it needs to render
  D: App depends on abstractions (protocols/base classes), not concretions
"""

from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Protocol
import json

from flask import Flask, render_template_string, request, jsonify


# ── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class Reference:
    title: str
    url: str


@dataclass
class MCPConcept:
    id: str
    title: str
    body: str
    references: list[Reference] = field(default_factory=list)


@dataclass
class QuizQuestion:
    id: int
    concept_id: str
    question: str
    choices: list[str]
    correct_index: int
    explanation: str
    reference: Reference


# ── Repositories (Open/Closed — extend by adding data, not modifying code) ───

class ConceptRepository:
    def all(self) -> list[MCPConcept]:
        return [
            MCPConcept(
                id="overview",
                title="1. What is MCP?",
                body="""
                    <p>The <strong>Model Context Protocol (MCP)</strong> is an open standard developed by Anthropic
                    that defines how AI models communicate with external tools, data sources, and services.
                    It provides a unified way for LLM hosts (like Claude Desktop or IDEs) to connect to
                    <em>MCP servers</em> that expose capabilities.</p>
                    <p>MCP is built on <strong>JSON-RPC 2.0</strong>, making it language-agnostic and easy
                    to implement. Every message is a JSON object with a method, parameters, and an id.</p>
                    <ul>
                        <li>Standardizes tool and resource discovery</li>
                        <li>Separates concerns between the model and its context sources</li>
                        <li>Enables reusable, composable server implementations</li>
                    </ul>
                """,
                references=[
                    Reference("MCP Introduction — Anthropic Docs", "https://modelcontextprotocol.io/introduction"),
                    Reference("JSON-RPC 2.0 Specification", "https://www.jsonrpc.org/specification"),
                ]
            ),
            MCPConcept(
                id="architecture",
                title="2. Client–Server–Host Architecture",
                body="""
                    <p>MCP defines three roles:</p>
                    <ul>
                        <li><strong>Host</strong> — The application the user runs (e.g., Claude Desktop, an IDE extension).
                            It manages one or more clients and routes model requests.</li>
                        <li><strong>Client</strong> — Lives inside the host. Maintains a 1-to-1 connection with a server
                            and handles the MCP protocol lifecycle.</li>
                        <li><strong>Server</strong> — An independent process or service that exposes Resources, Tools,
                            and Prompts to the client.</li>
                    </ul>
                    <p>This separation means the model never talks directly to external systems — all access is
                    mediated and auditable through the client/host layer.</p>
                """,
                references=[
                    Reference("MCP Architecture Overview", "https://modelcontextprotocol.io/docs/concepts/architecture"),
                ]
            ),
            MCPConcept(
                id="transport",
                title="3. Transport Mechanisms",
                body="""
                    <p>MCP supports two built-in transports:</p>
                    <ul>
                        <li><strong>stdio</strong> — The server is a subprocess; the client communicates over
                            stdin/stdout. Best for local servers. Simple, no networking required.</li>
                        <li><strong>HTTP + SSE (Server-Sent Events)</strong> — The server is a remote HTTP service.
                            The client sends requests via HTTP POST and receives responses/notifications over
                            a persistent SSE stream. Best for cloud-hosted servers.</li>
                    </ul>
                    <p>Custom transports are possible by implementing the transport interface, keeping
                    the protocol layer decoupled from the delivery mechanism.</p>
                """,
                references=[
                    Reference("MCP Transports", "https://modelcontextprotocol.io/docs/concepts/transports"),
                ]
            ),
            MCPConcept(
                id="resources",
                title="4. Resources",
                body="""
                    <p><strong>Resources</strong> expose read-only data to the model. Each resource is identified
                    by a URI (e.g., <code>file:///path/to/doc</code> or <code>db://users/42</code>).</p>
                    <p>Resources can be:</p>
                    <ul>
                        <li><strong>Static</strong> — Listed upfront and fetched on demand.</li>
                        <li><strong>Dynamic</strong> — Discovered via URI templates (e.g., <code>repo://{owner}/{repo}</code>).</li>
                    </ul>
                    <p>Resource contents are returned as text or binary blobs. The server can also push
                    notifications when resources change, allowing the host to re-read them.</p>
                """,
                references=[
                    Reference("MCP Resources", "https://modelcontextprotocol.io/docs/concepts/resources"),
                ]
            ),
            MCPConcept(
                id="tools",
                title="5. Tools",
                body="""
                    <p><strong>Tools</strong> are executable functions the model can invoke. Unlike resources
                    (read-only), tools can have side effects — writing files, calling APIs, running queries.</p>
                    <p>Each tool declares:</p>
                    <ul>
                        <li><strong>name</strong> — Unique identifier.</li>
                        <li><strong>description</strong> — Natural-language description for the model.</li>
                        <li><strong>inputSchema</strong> — JSON Schema describing the expected parameters.</li>
                    </ul>
                    <p>The model requests a tool call; the host/user may approve it; the server executes it
                    and returns a result. This keeps the model in control while the human stays in the loop.</p>
                """,
                references=[
                    Reference("MCP Tools", "https://modelcontextprotocol.io/docs/concepts/tools"),
                ]
            ),
            MCPConcept(
                id="prompts",
                title="6. Prompts",
                body="""
                    <p><strong>Prompts</strong> are reusable, parameterised prompt templates that servers expose
                    to clients. They encode best practices for interacting with a server's domain.</p>
                    <p>A prompt can accept arguments (e.g., <code>{ "language": "Python" }</code>) and returns
                    a list of messages ready to insert into a conversation.</p>
                    <p>This is different from tools — prompts shape the conversation context,
                    whereas tools perform actions.</p>
                """,
                references=[
                    Reference("MCP Prompts", "https://modelcontextprotocol.io/docs/concepts/prompts"),
                ]
            ),
            MCPConcept(
                id="sampling",
                title="7. Sampling",
                body="""
                    <p><strong>Sampling</strong> lets a server ask the client to perform an LLM completion
                    on its behalf. This enables agentic loops where the server can reason with the model
                    without needing its own model access.</p>
                    <p>The flow:</p>
                    <ol>
                        <li>Server sends a <code>sampling/createMessage</code> request to the client.</li>
                        <li>The host presents it to the user for approval (human-in-the-loop).</li>
                        <li>The client forwards the request to the model and returns the result.</li>
                    </ol>
                    <p>This is a key enabler for recursive/agentic MCP server designs.</p>
                """,
                references=[
                    Reference("MCP Sampling", "https://modelcontextprotocol.io/docs/concepts/sampling"),
                ]
            ),
            MCPConcept(
                id="lifecycle",
                title="8. Connection Lifecycle",
                body="""
                    <p>Every MCP connection follows a structured lifecycle:</p>
                    <ol>
                        <li><strong>Initialization</strong> — Client sends <code>initialize</code> with its
                            protocol version and capabilities. Server replies with its own capabilities.</li>
                        <li><strong>Initialized notification</strong> — Client confirms readiness.</li>
                        <li><strong>Operation</strong> — Normal request/response and notification exchange.</li>
                        <li><strong>Shutdown</strong> — Either side can close the connection cleanly.</li>
                    </ol>
                    <p>Capabilities negotiated at init time determine which features are available for
                    the session (e.g., whether the server supports subscriptions or sampling).</p>
                """,
                references=[
                    Reference("MCP Lifecycle", "https://modelcontextprotocol.io/docs/concepts/architecture#connection-lifecycle"),
                ]
            ),
            MCPConcept(
                id="roots",
                title="9. Roots",
                body="""
                    <p><strong>Roots</strong> tell the server which filesystem or URI locations the client
                    considers in scope. A client may declare roots like:</p>
                    <ul>
                        <li><code>file:///home/user/project</code></li>
                        <li><code>https://api.example.com/v1</code></li>
                    </ul>
                    <p>Servers use roots as hints for what to expose — they should avoid accessing paths
                    outside declared roots. Roots can change during a session via notifications,
                    allowing dynamic workspace changes.</p>
                """,
                references=[
                    Reference("MCP Roots", "https://modelcontextprotocol.io/docs/concepts/roots"),
                ]
            ),
        ]


class QuizRepository:
    def all(self) -> list[QuizQuestion]:
        return [
            QuizQuestion(
                id=1,
                concept_id="overview",
                question="What underlying protocol does MCP use for its message format?",
                choices=[
                    "REST / HTTP",
                    "GraphQL",
                    "JSON-RPC 2.0",
                    "gRPC / Protocol Buffers",
                ],
                correct_index=2,
                explanation="MCP is built on JSON-RPC 2.0, making every message a JSON object with a method, params, and id.",
                reference=Reference("MCP Introduction", "https://modelcontextprotocol.io/introduction"),
            ),
            QuizQuestion(
                id=2,
                concept_id="architecture",
                question="In the MCP architecture, which component maintains a 1-to-1 connection with the server?",
                choices=["The Host", "The Model", "The Client", "The Router"],
                correct_index=2,
                explanation="The Client (inside the Host) maintains a 1-to-1 persistent connection with an MCP server.",
                reference=Reference("MCP Architecture", "https://modelcontextprotocol.io/docs/concepts/architecture"),
            ),
            QuizQuestion(
                id=3,
                concept_id="transport",
                question="Which transport is best suited for a locally running MCP server subprocess?",
                choices=["HTTP + SSE", "WebSockets", "stdio", "gRPC"],
                correct_index=2,
                explanation="stdio transport uses stdin/stdout between parent and child process — ideal for local servers with no networking overhead.",
                reference=Reference("MCP Transports", "https://modelcontextprotocol.io/docs/concepts/transports"),
            ),
            QuizQuestion(
                id=4,
                concept_id="resources",
                question="How are MCP Resources primarily identified?",
                choices=["By a numeric integer ID", "By a URI (Uniform Resource Identifier)", "By a tool name", "By a JSON Schema"],
                correct_index=1,
                explanation="Each MCP Resource has a URI (e.g. file:///path or db://table/row) that uniquely identifies it.",
                reference=Reference("MCP Resources", "https://modelcontextprotocol.io/docs/concepts/resources"),
            ),
            QuizQuestion(
                id=5,
                concept_id="tools",
                question="What does a Tool's inputSchema field describe?",
                choices=[
                    "The return type of the tool",
                    "The model that should execute the tool",
                    "The expected parameters using JSON Schema",
                    "The transport layer to use",
                ],
                correct_index=2,
                explanation="inputSchema is a JSON Schema object that describes the parameters the tool expects, enabling validation and documentation.",
                reference=Reference("MCP Tools", "https://modelcontextprotocol.io/docs/concepts/tools"),
            ),
            QuizQuestion(
                id=6,
                concept_id="prompts",
                question="What is the primary difference between MCP Prompts and MCP Tools?",
                choices=[
                    "Prompts are faster to execute than Tools",
                    "Prompts shape conversation context; Tools perform actions with side effects",
                    "Tools are read-only; Prompts can write data",
                    "There is no difference — they are interchangeable",
                ],
                correct_index=1,
                explanation="Prompts are reusable message templates that set context. Tools are executable functions that perform actions and may have side effects.",
                reference=Reference("MCP Prompts", "https://modelcontextprotocol.io/docs/concepts/prompts"),
            ),
            QuizQuestion(
                id=7,
                concept_id="sampling",
                question="When a server uses MCP Sampling, who ultimately fulfils the LLM completion request?",
                choices=[
                    "The server itself using a local model",
                    "The MCP protocol layer",
                    "The client, which forwards it to the model via the host",
                    "A third-party cloud API called directly by the server",
                ],
                correct_index=2,
                explanation="The server sends sampling/createMessage to the client. The host can seek user approval, then forwards the request to the model and returns the result to the server.",
                reference=Reference("MCP Sampling", "https://modelcontextprotocol.io/docs/concepts/sampling"),
            ),
            QuizQuestion(
                id=8,
                concept_id="lifecycle",
                question="What is the first step in the MCP connection lifecycle?",
                choices=[
                    "The server sends a 'ready' notification",
                    "The client sends an 'initialize' request with its protocol version and capabilities",
                    "The model requests a list of available tools",
                    "The host opens a sampling session",
                ],
                correct_index=1,
                explanation="The lifecycle begins with the client sending 'initialize', declaring its protocol version and capabilities. The server replies with its own capabilities to complete negotiation.",
                reference=Reference("MCP Lifecycle", "https://modelcontextprotocol.io/docs/concepts/architecture#connection-lifecycle"),
            ),
            QuizQuestion(
                id=9,
                concept_id="roots",
                question="What is the purpose of MCP Roots?",
                choices=[
                    "To define the root JSON-RPC method namespace",
                    "To tell the server which filesystem or URI locations the client considers in scope",
                    "To set the root certificate for TLS connections",
                    "To declare the root tool that orchestrates all other tools",
                ],
                correct_index=1,
                explanation="Roots are URI hints from the client to the server indicating which locations are in scope. Servers should avoid accessing paths outside declared roots.",
                reference=Reference("MCP Roots", "https://modelcontextprotocol.io/docs/concepts/roots"),
            ),
        ]


# ── Evaluator (Single Responsibility — only checks answers) ──────────────────

class QuizEvaluator:
    def __init__(self, questions: list[QuizQuestion]) -> None:
        self._index: dict[int, QuizQuestion] = {q.id: q for q in questions}

    def evaluate(self, question_id: int, chosen_index: int) -> dict:
        q = self._index.get(question_id)
        if q is None:
            return {"error": "Question not found"}
        correct = chosen_index == q.correct_index
        return {
            "correct": correct,
            "correct_index": q.correct_index,
            "explanation": q.explanation,
            "reference_title": q.reference.title,
            "reference_url": q.reference.url,
        }


# ── Renderer Protocol (Dependency Inversion — depend on abstraction) ──────────

class PageRenderer(Protocol):
    def render(self, concepts: list[MCPConcept], questions: list[QuizQuestion]) -> str:
        ...


class HTMLPageRenderer:
    """Renders the full tutorial page as an HTML string. (Single Responsibility)"""

    def render(self, concepts: list[MCPConcept], questions: list[QuizQuestion]) -> str:
        concepts_html = self._render_concepts(concepts)
        quiz_html = self._render_quiz(questions)
        return TEMPLATE.replace("{{CONCEPTS}}", concepts_html).replace("{{QUIZ}}", quiz_html)

    def _render_concepts(self, concepts: list[MCPConcept]) -> str:
        parts = []
        for c in concepts:
            refs = "".join(
                f'<li><a href="{r.url}" target="_blank" rel="noopener">{r.title}</a></li>'
                for r in c.references
            )
            parts.append(f"""
            <section class="concept" id="{c.id}">
                <h2>{c.title}</h2>
                {c.body}
                <div class="references">
                    <h4>References</h4>
                    <ul>{refs}</ul>
                </div>
            </section>
            """)
        return "\n".join(parts)

    def _render_quiz(self, questions: list[QuizQuestion]) -> str:
        parts = []
        for q in questions:
            choices = "".join(
                f"""<label class="choice" data-index="{i}">
                    <input type="radio" name="q{q.id}" value="{i}">
                    <span>{choice}</span>
                </label>"""
                for i, choice in enumerate(q.choices)
            )
            parts.append(f"""
            <div class="quiz-question" data-qid="{q.id}">
                <p class="q-text"><strong>Q{q.id}.</strong> {q.question}</p>
                <div class="choices">{choices}</div>
                <button class="submit-btn" onclick="submitAnswer({q.id})">Submit</button>
                <div class="feedback" id="feedback-{q.id}"></div>
            </div>
            """)
        return "\n".join(parts)


# ── HTML Template ─────────────────────────────────────────────────────────────

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Important MCP Server Concepts</title>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2e3248;
    --accent: #7c5cfc;
    --accent2: #4fc3f7;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --success: #22c55e;
    --error: #f87171;
    --radius: 10px;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: system-ui, sans-serif;
         line-height: 1.7; font-size: 16px; }
  a { color: var(--accent2); }

  header {
    background: linear-gradient(135deg, #1a1d27 0%, #12162a 100%);
    border-bottom: 1px solid var(--border);
    padding: 2.5rem 2rem 2rem;
    text-align: center;
  }
  header h1 { font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  header p { color: var(--muted); margin-top: .5rem; font-size: 1rem; }

  nav {
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: .75rem 2rem; display: flex; gap: 1rem; flex-wrap: wrap;
    justify-content: center; position: sticky; top: 0; z-index: 10;
  }
  nav a { color: var(--muted); text-decoration: none; font-size: .85rem;
           padding: .25rem .6rem; border-radius: 6px; transition: background .2s, color .2s; }
  nav a:hover { background: var(--border); color: var(--text); }

  main { max-width: 860px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }

  .concept {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 1.75rem 2rem; margin-bottom: 2rem;
  }
  .concept h2 { font-size: 1.35rem; margin-bottom: 1rem; color: var(--accent2); }
  .concept p, .concept li { color: var(--text); margin-bottom: .5rem; }
  .concept ul, .concept ol { padding-left: 1.4rem; margin: .6rem 0 .6rem; }
  .concept code { background: #232638; border: 1px solid var(--border); border-radius: 4px;
                  padding: .1em .4em; font-size: .88em; color: #c8b1ff; }
  .references { margin-top: 1.25rem; padding-top: 1rem; border-top: 1px solid var(--border); }
  .references h4 { font-size: .8rem; text-transform: uppercase; letter-spacing: .08em;
                   color: var(--muted); margin-bottom: .4rem; }
  .references ul { padding-left: 1.2rem; }
  .references li { font-size: .88rem; }

  .quiz-section { margin-top: 3rem; }
  .quiz-section > h2 { font-size: 1.7rem; font-weight: 700; margin-bottom: .4rem;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .quiz-section > p { color: var(--muted); margin-bottom: 1.5rem; }

  .quiz-question {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 1.5rem; margin-bottom: 1.5rem;
  }
  .q-text { margin-bottom: 1rem; font-size: 1rem; }
  .choices { display: flex; flex-direction: column; gap: .6rem; margin-bottom: 1rem; }
  .choice {
    display: flex; align-items: flex-start; gap: .7rem; padding: .65rem 1rem;
    border: 1px solid var(--border); border-radius: 8px; cursor: pointer;
    transition: border-color .2s, background .2s;
  }
  .choice:hover { border-color: var(--accent); background: #231f38; }
  .choice input { margin-top: .2rem; accent-color: var(--accent); cursor: pointer; }
  .choice.correct-choice { border-color: var(--success); background: #0f2a1a; }
  .choice.wrong-choice { border-color: var(--error); background: #2a1010; }

  .submit-btn {
    background: var(--accent); color: #fff; border: none; padding: .55rem 1.4rem;
    border-radius: 7px; font-size: .92rem; font-weight: 600; cursor: pointer;
    transition: opacity .2s;
  }
  .submit-btn:hover { opacity: .85; }
  .submit-btn:disabled { opacity: .4; cursor: default; }

  .feedback { margin-top: 1rem; padding: 1rem 1.2rem; border-radius: 8px; font-size: .9rem;
              display: none; }
  .feedback.correct { background: #0d2a1a; border: 1px solid var(--success); color: var(--success); }
  .feedback.incorrect { background: #2a0d0d; border: 1px solid var(--error); color: var(--text); }
  .feedback .feedback-title { font-weight: 700; font-size: 1rem; margin-bottom: .4rem; }
  .feedback .ref-link { color: var(--accent2); font-size: .85rem; margin-top: .5rem; display: block; }

  #score-banner {
    display: none; background: var(--surface); border: 1px solid var(--accent);
    border-radius: var(--radius); padding: 1.5rem 2rem; text-align: center; margin-top: 2rem;
  }
  #score-banner h3 { font-size: 1.4rem; margin-bottom: .4rem; color: var(--accent2); }
  #score-banner p { color: var(--muted); }
  #score-banner .score-num { font-size: 2.5rem; font-weight: 800; color: var(--accent); }
</style>
</head>
<body>

<header>
  <h1>Important MCP Server Concepts</h1>
  <p>A developer's guide to the Model Context Protocol — architecture, primitives, and lifecycle</p>
</header>

<nav>
  <a href="#overview">Overview</a>
  <a href="#architecture">Architecture</a>
  <a href="#transport">Transport</a>
  <a href="#resources">Resources</a>
  <a href="#tools">Tools</a>
  <a href="#prompts">Prompts</a>
  <a href="#sampling">Sampling</a>
  <a href="#lifecycle">Lifecycle</a>
  <a href="#roots">Roots</a>
  <a href="#quiz">Quiz</a>
</nav>

<main>
  {{CONCEPTS}}

  <section class="quiz-section" id="quiz">
    <h2>Knowledge Quiz</h2>
    <p>Test your understanding of MCP server concepts. Wrong answers show the correct answer and a reference.</p>
    {{QUIZ}}
    <div id="score-banner">
      <h3>Quiz Complete</h3>
      <div class="score-num" id="score-display"></div>
      <p id="score-msg"></p>
    </div>
  </section>
</main>

<script>
  const totalQuestions = document.querySelectorAll('.quiz-question').length;
  let answered = 0;
  let correct = 0;

  async function submitAnswer(qid) {
    const radios = document.querySelectorAll(`input[name="q${qid}"]`);
    let chosen = -1;
    radios.forEach((r, i) => { if (r.checked) chosen = i; });
    if (chosen === -1) { alert("Please select an answer."); return; }

    const btn = document.querySelector(`.quiz-question[data-qid="${qid}"] .submit-btn`);
    btn.disabled = true;
    radios.forEach(r => r.disabled = true);

    const res = await fetch('/check', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question_id: qid, chosen_index: chosen})
    });
    const data = await res.json();

    const fb = document.getElementById(`feedback-${qid}`);
    const choices = document.querySelectorAll(`.quiz-question[data-qid="${qid}"] .choice`);

    if (data.correct) {
      correct++;
      choices[chosen].classList.add('correct-choice');
      fb.className = 'feedback correct';
      fb.innerHTML = `<div class="feedback-title">✓ Correct!</div><p>${data.explanation}</p>
        <a class="ref-link" href="${data.reference_url}" target="_blank">Reference: ${data.reference_title}</a>`;
    } else {
      choices[chosen].classList.add('wrong-choice');
      choices[data.correct_index].classList.add('correct-choice');
      fb.className = 'feedback incorrect';
      fb.innerHTML = `<div class="feedback-title" style="color:var(--error)">✗ Incorrect</div>
        <p><strong>Correct answer:</strong> ${choices[data.correct_index].querySelector('span').textContent}</p>
        <p style="margin-top:.5rem">${data.explanation}</p>
        <a class="ref-link" href="${data.reference_url}" target="_blank">Reference: ${data.reference_title}</a>`;
    }
    fb.style.display = 'block';

    answered++;
    if (answered === totalQuestions) showScore();
  }

  function showScore() {
    const pct = Math.round((correct / totalQuestions) * 100);
    const msg = pct === 100 ? "Perfect score — MCP expert!" :
                pct >= 80  ? "Great work! A few concepts to review." :
                pct >= 60  ? "Good effort. Revisit the sections above." :
                             "Keep studying — the tutorial is right above!";
    document.getElementById('score-display').textContent = `${correct} / ${totalQuestions}`;
    document.getElementById('score-msg').textContent = msg;
    document.getElementById('score-banner').style.display = 'block';
    document.getElementById('score-banner').scrollIntoView({behavior: 'smooth'});
  }
</script>
</body>
</html>
"""


# ── App Factory (Dependency Inversion — wired here, not inside classes) ───────

def create_app() -> Flask:
    app = Flask(__name__)

    concept_repo = ConceptRepository()
    quiz_repo = QuizRepository()
    evaluator = QuizEvaluator(quiz_repo.all())
    renderer: PageRenderer = HTMLPageRenderer()

    page_html = renderer.render(concept_repo.all(), quiz_repo.all())

    @app.get("/")
    def index():
        return page_html

    @app.post("/check")
    def check():
        body = request.get_json(force=True)
        result = evaluator.evaluate(
            question_id=int(body["question_id"]),
            chosen_index=int(body["chosen_index"]),
        )
        return jsonify(result)

    return app


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = create_app()
    print("MCP Tutorial running at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
