"""vllm_down in devlab_perf.sh must stop the vLLM server by its port too: the pid it records can be a launcher that already exited,
and the next engine's start (the no-draft c4 row of the MXFP4 card job) then found the old server still answering."""
import re
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "devlab_perf.sh"


class VllmDown(unittest.TestCase):
    def test_stops_by_port_and_engine_core(self):
        s = SCRIPT.read_text()
        body = re.search(r"^vllm_down\(\) \{.*?^\}", s, re.S | re.M).group(0)
        self.assertIn('pkill -TERM -f "vllm.entrypoints.openai.api_server.*--port $VPORT', body)
        self.assertIn("pkill -9 -f", body)
        self.assertIn("VLLM::EngineCore", body)


if __name__ == "__main__":
    unittest.main()
