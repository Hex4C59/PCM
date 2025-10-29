import  csv
import os

class Logger:

    def __init__(self, log_dir, filename, console):
        os.makedirs(log_dir, exist_ok=True)

        self.log_path = os.path.join(log_dir, filename)
        self.console_enabled = console

        self.file = open(self.log_path, 'w', encoding='utf-8', newline='')
        self.writer = csv.writer(self.file)

        self.header_written = False
        self.console(f"Logger initialized. Logging metrics to: {self.log_path}")

    def log(self, data):
        if not self.header_written:
            self.writer.writerow(data.keys())
            self.header_written = True

        self.writer.writerow(data.values())

        self.file.flush()
        if self.console_enabled:
            message_parts = []
            for key, value in data.items():
                if key == 'lr':
                    message_parts.append(f"{key}: {value:.2g}")
                elif isinstance(value, float):
                    # 将浮点数格式化为保留4位小数
                    message_parts.append(f"{key}: {value:.4f}")
                else:
                    message_parts.append(f"{key}: {value}")
            
            message = " | ".join(message_parts)
            print(message)

    def console(self, message: str) -> None:
        if self.console_enabled:
            print(message)

    def close(self) -> None:
        if self.file:
            self.file.close()
            self.console("Logger closed.")
