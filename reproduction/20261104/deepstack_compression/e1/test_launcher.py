import unittest
from unittest.mock import patch
import sys
import types


# Import without invoking Docker or model loading.
run_panel=types.ModuleType('run_panel')
run_panel.container_command=lambda *x: None
run_panel.gpu_pids=lambda: []
run_panel.identity=lambda x: None
with patch.dict(sys.modules,{'run_panel':run_panel}):
    import launch


class LauncherExitTests(unittest.TestCase):
    def test_early_client_exit_cannot_be_success(self):
        state={'timed_out':False,'interrupted':None,'returncode':0,'client_exited_while_container_running':True}
        self.assertNotEqual(launch.exit_code(state),0)
    def test_interrupted_even_if_client_zero_is_failure(self):
        state={'timed_out':False,'interrupted':15,'returncode':0,'client_exited_while_container_running':False}
        self.assertNotEqual(launch.exit_code(state),0)
    def test_clean_exit(self):
        state={'timed_out':False,'interrupted':None,'returncode':0,'client_exited_while_container_running':False}
        self.assertEqual(launch.exit_code(state),0)

if __name__=='__main__': unittest.main()
