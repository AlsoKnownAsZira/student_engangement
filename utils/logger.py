"""
Logging utilities
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import json


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for terminal output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # Format the message
        record.levelname = f"{log_color}{record.levelname}{reset_color}"
        return super().format(record)


def setup_logger(name, log_file=None, level=logging.INFO, use_colors=True):
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        use_colors: Use colored output for console
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if use_colors:
        console_format = ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
    
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


class ExperimentLogger:
    """Logger for experiment tracking"""
    
    def __init__(self, output_dir, experiment_name):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_id = f"{experiment_name}_{timestamp}"
        
        # Create run directory
        self.run_dir = self.output_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        log_file = self.run_dir / 'log.txt'
        self.logger = setup_logger(
            f"experiment_{self.run_id}",
            log_file=log_file
        )
        
        # Metrics storage
        self.metrics = {
            'start_time': timestamp,
            'experiment_name': experiment_name,
            'config': {},
            'results': {}
        }
        
        self.logger.info(f"Experiment: {experiment_name}")
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info(f"Output directory: {self.run_dir}")
    
    def log_config(self, config):
        """Log configuration"""
        self.metrics['config'] = config
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_metric(self, name, value, step=None):
        """Log a metric"""
        if step is not None:
            self.logger.info(f"Step {step} | {name}: {value}")
        else:
            self.logger.info(f"{name}: {value}")
        
        if name not in self.metrics['results']:
            self.metrics['results'][name] = []
        
        self.metrics['results'][name].append({
            'value': value,
            'step': step,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_metrics(self, metrics_dict, step=None):
        """Log multiple metrics"""
        for name, value in metrics_dict.items():
            self.log_metric(name, value, step)
    
    def save_results(self, filename='results.json'):
        """Save results to JSON file"""
        self.metrics['end_time'] = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        output_path = self.run_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to: {output_path}")
        return output_path
    
    def get_output_path(self, filename):
        """Get path for output file in run directory"""
        return self.run_dir / filename


class ProgressLogger:
    """Simple progress logger for long operations"""
    
    def __init__(self, total, desc="Processing", logger=None):
        self.total = total
        self.current = 0
        self.desc = desc
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = datetime.now()
    
    def update(self, n=1):
        """Update progress"""
        self.current += n
        
        if self.current % max(1, self.total // 20) == 0 or self.current == self.total:
            progress = self.current / self.total * 100
            elapsed = (datetime.now() - self.start_time).total_seconds()
            
            if self.current > 0:
                eta = elapsed / self.current * (self.total - self.current)
                self.logger.info(
                    f"{self.desc}: {self.current}/{self.total} "
                    f"({progress:.1f}%) | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s"
                )
            else:
                self.logger.info(
                    f"{self.desc}: {self.current}/{self.total} ({progress:.1f}%)"
                )
    
    def finish(self):
        """Finish progress logging"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(
            f"{self.desc}: Completed {self.total} items in {elapsed:.1f}s "
            f"({self.total / elapsed:.1f} items/s)"
        )