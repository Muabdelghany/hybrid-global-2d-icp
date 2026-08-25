"""Pipeline orchestration for sequential DTPM module execution."""

import time
import logging

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates sequential execution of DTPM modules."""

    def __init__(self, config, data_manager=None):
        """
        Initialize pipeline.

        Args:
            config: SimulationConfig instance.
            data_manager: DataManager instance for result I/O.
        """
        self.config = config
        self.data_manager = data_manager
        self.state = {}  # Shared state dict accumulating module outputs
        self.modules = []
        self.execution_log = []

    def add_module(self, name, module_func, depends_on=None):
        """
        Register a module for execution.

        Args:
            name: Module identifier string.
            module_func: Callable that takes (state, config) and returns dict of results.
            depends_on: List of module names that must run before this one.
        """
        self.modules.append({
            'name': name,
            'func': module_func,
            'depends_on': depends_on or [],
        })

    def run(self, stop_after=None):
        """
        Execute all registered modules in order.

        Args:
            stop_after: If provided, stop after this module name.
        """
        for module in self.modules:
            name = module['name']
            logger.info(f"Running module: {name}")

            t0 = time.time()
            try:
                results = module['func'](self.state, self.config)
                if results:
                    self.state.update(results)
                elapsed = time.time() - t0
                self.execution_log.append({
                    'module': name, 'status': 'success', 'time': elapsed
                })
                logger.info(f"  {name} completed in {elapsed:.2f}s")
            except Exception as e:
                elapsed = time.time() - t0
                self.execution_log.append({
                    'module': name, 'status': 'failed', 'error': str(e), 'time': elapsed
                })
                logger.error(f"  {name} FAILED after {elapsed:.2f}s: {e}")
                raise

            if stop_after and name == stop_after:
                logger.info(f"Pipeline stopped after {name}")
                break

        return self.state

    def print_summary(self):
        """Print execution summary."""
        print("\n=== Pipeline Execution Summary ===")
        total = 0
        for entry in self.execution_log:
            status = "OK" if entry['status'] == 'success' else "FAIL"
            print(f"  [{status}] {entry['module']}: {entry['time']:.2f}s")
            total += entry['time']
        print(f"  Total: {total:.2f}s")
        print("=" * 35)
