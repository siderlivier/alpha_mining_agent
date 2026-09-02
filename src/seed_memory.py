"""
M2：初始化記憶結構（極簡骨架版）。

設計決策（使用者要求）：不預載 FinLab 文章或前置專案的結論——
避免把過多預設理論當成記憶餵給模型、汙染它自己的歸納。
agent 的所有經驗一律來自它親手跑過的 attempt 與證據編號。

本腳本只建立空結構，可重複執行（冪等，不會覆蓋既有內容）。

執行：python src/seed_memory.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory import Memory


def main():
    mem = Memory()
    mem.ensure()
    print(f"✅ 記憶結構就緒：{mem.root}")
    print(f"   attempts/         {len(list(mem.attempts_dir.glob('A-*.json')))} 筆")
    print(f"   library.json      {len(mem.own_library())} 個因子"
          f"（另有 {len(mem.reference_library())} 個參考因子）")
    print(f"   learnings.md      {len(mem.read_learnings())} 字元")


if __name__ == "__main__":
    main()
