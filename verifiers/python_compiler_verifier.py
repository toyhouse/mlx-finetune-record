import os
import py_compile
import tempfile

# verifier imports
from verifiers.base_verifier import BaseVerifier

class CodeCompileVerifier(BaseVerifier):
    """
    Simple verifier that checks if generated Python code:
      1) Compiles successfully.
      2) Defines a function called 'sum_of_list' that correctly
         sums a list of integers.
    """

    def get_reward(self, code_str: str) -> float:
        reward = 0.0

        # Attempt to compile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(code_str.encode("utf-8"))

        try:
            py_compile.compile(tmp_path, doraise=True)
            reward += 0.5  # Partial reward for compilation
        except Exception:
            # Compilation failed, return reward earned so far (0)
            os.remove(tmp_path)
            return reward

        # Check for 'sum_of_list' correctness
        try:
            local_ns = {}
            with open(tmp_path, "r") as f:
                code_content = f.read()
            exec(code_content, local_ns)

            if "sum_of_list" in local_ns:
                test_result = local_ns["sum_of_list"]([1, 2, 3])
                if test_result == 6:
                    reward += 0.5
        except Exception:
            pass
        finally:
            os.remove(tmp_path)

        return reward