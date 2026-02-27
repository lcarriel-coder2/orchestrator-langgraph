import os
import subprocess
import argparse
import hashlib
from dotenv import load_dotenv

# LangGraph & Checkpointing
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_anthropic import ChatAnthropic

# Importar el estado (Asegúrate de tener state.py en la misma carpeta)
from state import AgentState

load_dotenv()


def detect_language(text: str) -> str:
    """Detecta el lenguaje principal del prompt de forma muy simple."""
    lower = text.lower()
    if "javascript" in lower or " js" in lower:
        return "javascript"
    if "python" in lower or " py" in lower:
        return "python"
    # Por defecto asumimos Python si no se especifica nada
    return "python"


# --- CONFIGURACIÓN DEL MODELO ---
# Usamos Haiku: inteligente, rápido y económico para un flujo de agentes.
model = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)

# --- NODOS DEL SISTEMA ---


def check_epic_node(state: AgentState):
    print("\n--- 🔍 [1/6] REVISANDO EPIC ---")
    return {"execution_log": state['execution_log'] + ["Epic verificado."]}

def create_story_node(state: AgentState):
    print("--- 📝 [2/6] GENERANDO USER STORY ---")
    prompt = f"Genera una User Story técnica para: {state['input_prompt']}. Incluye criterios de aceptación."
    res = model.invoke(prompt)
    return {
        "current_story_id": "STORY-101",
        "input_prompt": res.content,
        "new_story_created": True,
        "execution_log": state['execution_log'] + ["User Story generada."]
    }

def party_mode_node(state: AgentState):
    print("--- 🎉 [3/6] PARTY MODE: REFINAMIENTO ---")
    # Debate simulado para mejorar la calidad técnica
    feedback = model.invoke(
        f"Actúa como Arquitecto. Refina esta story para que sea técnicamente perfecta: {state['input_prompt']}"
    )
    return {
        "input_prompt": feedback.content,
        "execution_log": state['execution_log'] + ["Story refinada por expertos."]
    }

def dev_story_node(state: AgentState):
    # Verificamos si hay un error previo en el log para activar el modo REPARACIÓN
    last_log = state['execution_log'][-1] if state['execution_log'] else ""
    is_repair = "ERROR QA" in last_log

    print(f"--- 💻 [4/6] DESARROLLO: {'REPARANDO' if is_repair else 'ESCRIBIENDO'} CÓDIGO ---")

    context = f"\nFALLO PREVIO: {last_log}\nCorrige el error anterior." if is_repair else ""

    # Determinar lenguaje objetivo: si ya existe en el estado (reparación), lo respetamos;
    # en caso contrario lo inferimos del prompt original.
    language = state.get("code_language") or detect_language(state["input_prompt"])
    language_label = "JavaScript" if language == "javascript" else "Python"

    prompt = (
        f"{context}\nEscribe el código {language_label} para: {state['input_prompt']}."
        " Responde SOLO el código, sin texto ni markdown."
    )

    res = model.invoke(prompt)
    content = res.content.strip()

    # Eliminar fences de código genéricos ```lang ... ```
    if content.startswith("```"):
        # Quitar el primer ```xxx
        lines = content.splitlines()
        if lines:
            # eliminar la primera línea (``` o ```python, ```javascript, etc.)
            lines = lines[1:]
        # Eliminar posible cierre ```
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    code = content

    folder = f"project_{state['team_id']}"
    os.makedirs(folder, exist_ok=True)
    extension = "js" if language == "javascript" else "py"
    file_path = os.path.join(folder, f"main.{extension}")

    with open(file_path, "w") as f:
        f.write(code)

    return {
        "branch_name": f"bmad/feat-{state['team_id']}-101",
        "code_language": language,
        "code_filepath": file_path,
        "execution_log": state['execution_log'] + [f"Código guardado en {file_path}"]
    }

def qa_automation_node(state: AgentState):
    print("--- 🧪 [5/6] QA: VALIDANDO CÓDIGO ---")
    language = state.get("code_language") or detect_language(state["input_prompt"])
    default_extension = "js" if language == "javascript" else "py"
    file_path = state.get("code_filepath") or f"project_{state['team_id']}/main.{default_extension}"

    # QA específico por lenguaje
    if language == "python":
        # Intenta compilar el archivo para detectar errores de sintaxis
        result = subprocess.run(
            ["python3", "-m", "py_compile", file_path],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("✅ QA: Código Python válido.")
            return {
                "failure_state": False,
                "qa_results": {
                    "status": "passed",
                    "language": "python",
                    "details": "Compilación exitosa con py_compile.",
                    "file": file_path,
                },
                "execution_log": state["execution_log"] + ["QA exitoso (Python)."],
            }
        else:
            error_msg = result.stderr.strip()
            print("❌ QA FALLIDO (Python): Error detectado.")
            return {
                "failure_state": True,
                "qa_results": {
                    "status": "failed",
                    "language": "python",
                    "details": error_msg,
                    "file": file_path,
                },
                "execution_log": state["execution_log"] + [f"ERROR QA: {error_msg}"],
            }
    else:
        # Para JavaScript intentamos una verificación básica con node --check
        try:
            result = subprocess.run(
                ["node", "--check", file_path],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            # Node no está instalado: lo registramos pero no bloqueamos el flujo
            msg = "Node.js no encontrado en el entorno; se omite QA de JavaScript."
            print(f"⚠️ {msg}")
            return {
                "failure_state": False,
                "qa_results": {
                    "status": "skipped",
                    "language": "javascript",
                    "details": msg,
                    "file": file_path,
                },
                "execution_log": state["execution_log"] + [msg],
            }

        if result.returncode == 0:
            print("✅ QA: Código JavaScript válido (node --check).")
            return {
                "failure_state": False,
                "qa_results": {
                    "status": "passed",
                    "language": "javascript",
                    "details": "Validación exitosa con node --check.",
                    "file": file_path,
                },
                "execution_log": state["execution_log"] + ["QA exitoso (JavaScript)."],
            }
        else:
            error_msg = result.stderr.strip()
            print("❌ QA FALLIDO (JavaScript): Error detectado.")
            return {
                "failure_state": True,
                "qa_results": {
                    "status": "failed",
                    "language": "javascript",
                    "details": error_msg,
                    "file": file_path,
                },
                "execution_log": state["execution_log"] + [f"ERROR QA: {error_msg}"],
            }

def github_automation_node(state: AgentState):
    print("--- 🚀 [6/6] GITHUB: COMMIT Y BRANCH ---")
    branch = state['branch_name']
    try:
        # Operaciones Git locales en WSL2
        subprocess.run(["git", "checkout", "-b", branch], capture_output=True)

        # Solo añadimos al commit el archivo de código generado
        code_filepath = state.get("code_filepath") or f"project_{state['team_id']}/main.py"
        subprocess.run(["git", "add", code_filepath], check=True)
        subprocess.run(
            ["git", "commit", "-m", f"feat({state['team_id']}): implement story {state.get('current_story_id', 'N/A')}"],
            check=True,
        )

        # Obtener el SHA del commit recién creado
        sha_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        commit_sha = sha_result.stdout.strip() if sha_result.returncode == 0 else None

        print(f"✅ Rama '{branch}' creada con commit local.")
        return {
            "pr_url": "https://github.com/simulado/pull/1",
            "commit_sha": commit_sha,
            "execution_log": state['execution_log'] + ["Proceso Git finalizado."]
        }
    except Exception as e:
        print(f"⚠️ Git Error: Asegúrate de ejecutar 'git init' primero. Detalle: {e}")
        return {"failure_state": True}

# --- LÓGICA DE CONTROL (ROUTER) ---

def router_logic(state: AgentState):
    """Controla el flujo: Éxito, Reintento (Máx 3) o Aborto."""
    if not state["failure_state"]:
        return "success"
    
    # 🛡️ PROTECCIÓN DE LOOP: Máximo 3 intentos
    attempts = sum(1 for log in state['execution_log'] if "ERROR QA" in log)
    
    if attempts >= 3:
        print(f"\n❌ [ABORTADO] Se alcanzó el límite de {attempts} intentos. Revisa los logs.")
        return "abort"
    
    print(f"\n🔄 [REINTENTO {attempts + 1}/3] El código tiene errores. Reenviando al desarrollador...")
    return "repair"

# --- CONSTRUCCIÓN DEL GRAFO ---

workflow = StateGraph(AgentState)

workflow.add_node("check_epic", check_epic_node)
workflow.add_node("create_story", create_story_node)
workflow.add_node("party_mode", party_mode_node)
workflow.add_node("dev_story", dev_story_node)
workflow.add_node("qa_automate", qa_automation_node)
workflow.add_node("github_push", github_automation_node)

workflow.set_entry_point("check_epic")
workflow.add_edge("check_epic", "create_story")
workflow.add_edge("create_story", "party_mode")
workflow.add_edge("party_mode", "dev_story")
workflow.add_edge("dev_story", "qa_automate")

workflow.add_conditional_edges(
    "qa_automate",
    router_logic,
    {
        "success": "github_push",
        "repair": "dev_story",
        "abort": END
    }
)

workflow.add_edge("github_push", END)






# --- INTERFAZ CLI ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMAD Autonomous Agent CLI")
    parser.add_argument("--team_id", type=str, required=True, help="ID del equipo")
    parser.add_argument("--prompt", type=str, required=True, help="Requerimiento de software")
    
    args = parser.parse_args()

    # Thread ID único por cada prompt para no mezclar memorias en SQLite
    task_hash = hashlib.md5(args.prompt.encode()).hexdigest()[:8]
    thread_id = f"{args.team_id}_{task_hash}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "team_id": args.team_id,
        "input_prompt": args.prompt,
        "current_epic_id": None,
        "current_story_id": None,
        "branch_name": None,
        "commit_sha": None,
        "pr_url": None,
        "code_language": None,
        "code_filepath": None,
        "code_review_issues": [],
        "new_story_created": False,
        "qa_results": {},
        "failure_state": False,
        "execution_log": [],
    }

    print(f"\n🚀 BMAD CLI INICIADO | Equipo: {args.team_id} | Session: {thread_id}")
    print("-" * 60)

    with SqliteSaver.from_conn_string("checkpoints.sqlite") as memory:
        app = workflow.compile(checkpointer=memory)
        for output in app.stream(initial_state, config=config):
            for node, _ in output.items():
                print(f"✔️ Nodo '{node}' finalizado.")

    print("-" * 60)
    print("✅ Operación terminada.")