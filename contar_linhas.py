import os
from collections import Counter

# Se tiver tiktoken instalado, usamos ele. Caso contrário, fazemos aproximação.
try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")  # mesmo do GPT-4/5
    def contar_tokens(texto: str) -> int:
        return len(enc.encode(texto))
except ImportError:
    print("⚠️ tiktoken não encontrado, usando aproximação de 6 tokens/linha")
    def contar_tokens(texto: str) -> int:
        return sum(len(l.strip().split()) for l in texto.splitlines())

IGNORAR = {}

total_linhas = 0
total_tokens = 0
arquivos_contados = 0
dir_counter = Counter()

for root, dirs, files in os.walk(".", topdown=True):
    parts = root.split(os.sep)

    if any(
        part in IGNORAR
        or part.lower().endswith("env")
        or (part.startswith(".") and part != ".")
        for part in parts
    ):
        dirs[:] = []
        continue

    dirs[:] = [
        d for d in dirs
        if d not in IGNORAR
           and not d.lower().endswith("env")
           and not (d.startswith(".") and d != ".")
    ]

    py_files = [f for f in files if f.endswith(".py")]
    if py_files:
        dir_counter[root] += len(py_files)

    for file in py_files:
        caminho = os.path.join(root, file)
        try:
            with open(caminho, "r", encoding="utf-8") as f:
                conteudo = f.read()
                linhas = conteudo.count("\n") + 1
                tokens = contar_tokens(conteudo)
                total_linhas += linhas
                total_tokens += tokens
                arquivos_contados += 1
        except Exception as e:
            print(f"Erro ao ler {caminho}: {e}")

print(f"\n📁 {arquivos_contados} arquivos Python encontrados")
print(f"📊 {total_linhas:,} linhas totais de código Python")
print(f"🔢 {total_tokens:,} tokens estimados (GPT-5)\n")

print("🗂️  Diretórios com mais arquivos Python:")
for path, count in dir_counter.most_common(10):
    print(f"  {count:4d} arquivos em {path}")
