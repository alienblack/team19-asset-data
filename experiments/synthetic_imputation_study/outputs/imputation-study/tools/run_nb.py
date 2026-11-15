"""Executes a notebook in the 'imputation' kernel and writes its outputs back to the same file.

Usage: run_nb.py NOTEBOOK [WORKING_DIRECTORY]
"""
import json
import queue
import sys
import time
from pathlib import Path

from jupyter_client.manager import start_new_kernel

sys.stdout.reconfigure(encoding="utf-8")
path = Path(sys.argv[1])
cwd = Path(sys.argv[2]) if len(sys.argv) > 2 else path.parent
notebook = json.loads(path.read_text(encoding="utf-8"))
manager, client = start_new_kernel(kernel_name="imputation", cwd=str(cwd))
failed, count = False, 0
try:
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        count += 1
        cell["outputs"], cell["execution_count"] = [], count
        began = time.time()
        print(f"\n===== cell {index} =====", flush=True)
        message_id = client.execute("".join(cell["source"]))
        while True:
            try:
                message = client.get_iopub_msg(timeout=3000)
            except queue.Empty:
                print("TIMEOUT")
                failed = True
                break
            if message["parent_header"].get("msg_id") != message_id:
                continue
            kind, content = message["msg_type"], message["content"]
            if kind == "stream":
                print(content["text"], end="", flush=True)
                previous = cell["outputs"][-1] if cell["outputs"] else None
                if previous and previous["output_type"] == "stream" and previous["name"] == content["name"]:
                    previous["text"] += content["text"]
                else:
                    cell["outputs"].append({"output_type": "stream", "name": content["name"], "text": content["text"]})
            elif kind == "display_data":
                print(content["data"].get("text/plain", "")[:3000], flush=True)
                cell["outputs"].append({"output_type": "display_data", "data": content["data"], "metadata": content["metadata"]})
            elif kind == "execute_result":
                print(content["data"].get("text/plain", "")[:3000], flush=True)
                cell["outputs"].append({"output_type": "execute_result", "data": content["data"],
                                        "metadata": content["metadata"], "execution_count": count})
            elif kind == "error":
                print("\n".join(content["traceback"]), flush=True)
                cell["outputs"].append({"output_type": "error", "ename": content["ename"],
                                        "evalue": content["evalue"], "traceback": content["traceback"]})
                failed = True
            elif kind == "status" and content["execution_state"] == "idle":
                break
        print(f"[cell {index}: {time.time() - began:.1f}s]", flush=True)
        if failed:
            break
finally:
    client.stop_channels()
    manager.shutdown_kernel(now=True)
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
print("\nRESULT:", "FAILED" if failed else "ALL CELLS PASSED")
