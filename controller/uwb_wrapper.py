import serial
import threading
import time
from typing import Tuple
import difflib

class UWBWrapper:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, callback=None):
        self.port = port
        self.baudrate = baudrate
        self.serial_port = None
        self.latest_position = (0.0, 0.0, 0.0)
        self.running = False
        self.thread = None
        self.callback = callback

        # anchor settings
        self.anchor_count = 0
        self.anchors = []
        self.anchors_sent = False

    @staticmethod
    def fuzzy_match(target, candidate, threshold=0.7):
        similarity = difflib.SequenceMatcher(None, target, candidate).ratio()
        return similarity >= threshold

    def start(self):
        self.serial_port = serial.Serial(self.port, self.baudrate, timeout=1)
        self.reset_anchors()
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        print("UWBWrapper: Serial port connected and reading thread started.")
        

    def start_with_retry(self):
        def try_connect():
            while not self.running:
                try:
                    print("UWBWrapper: Trying to start UWB...")
                    self.start()
                    print("UWBWrapper: UWB started successfully.")
                except Exception as e:
                    print(f"UWBWrapper: Failed to open serial port: {e}")
                    time.sleep(1)
        threading.Thread(target=try_connect, daemon=True).start()

    def register_callback(self, cb):
        self.callback = cb

    def stop(self):
        self.running = False
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            print("UWBWrapper: Serial port closed.")

    def _read_loop(self):
        buffer = ""
        while self.running:
            try:
                if self.serial_port.in_waiting:
                    data = self.serial_port.read(self.serial_port.in_waiting).decode(errors='ignore')
                    buffer += data

                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if not line:
                            continue
                      # print(f"[DEBUG] Raw line: {repr(line)}")

                        # 判斷 anchor 送成功
                        if not self.anchors_sent:
                            if self.fuzzy_match("Delivery Success", line) or self.fuzzy_match("Start Positioning!", line):
                                print("UWBWrapper: Anchor delivery successful.")
                                self.anchors_sent = True
                                continue
                            elif self.fuzzy_match("Delivery Fail", line):
                                print("UWBWrapper: Anchor delivery failed.")
                                continue
                            else:
                                print("[DEBUG] Anchors not sent yet, skipping line.")
                                continue

                        parts = line.split(',')
                        if len(parts) == 3:
                            try:
                                x, y, z = map(float, parts)
                                self.latest_position = (x, y, z)
                                if self.callback:
                                  # print(f"[UWBWrapper] Callback: {x}, {y}, {z}")
                                    self.callback((time.time(), x, y, z))
                            except ValueError:
                                print(f"[UWBWrapper] Invalid float values: {line}")
                        else:
                            print(f"[UWBWrapper] Unrecognized line: {line}")
            except Exception as e:
                print(f"UWBWrapper: Error parsing line: {e}")
            time.sleep(0.01)

    def get_user_position(self) -> Tuple[float, float, float]:
        return self.latest_position

    def set_anchor_count(self, n: int):
        if n <= 0:
            print("UWBWrapper: Anchor count must be positive.")
            return
        self.anchor_count = n
        self.anchors = [None] * n
        print(f"UWBWrapper: Anchor count set to {n}")

    def input_anchor_line(self, line_str: str):
        parts = line_str.strip().split(',')
        if len(parts) != 4:
            print("UWBWrapper: Invalid anchor input format.")
            return
        try:
            i = int(parts[0]) - 1
            x, y, z = map(float, parts[1:])
        except Exception:
            print("UWBWrapper: Failed to parse anchor line.")
            return

        if i < 0 or i >= self.anchor_count:
            print("UWBWrapper: Anchor index out of range.")
            return

        self.anchors[i] = (x, y, z)
        print(f"[Anchor{i+1}] Position set to x={x}, y={y}, z={z}")

        if all(a is not None for a in self.anchors):
            print("UWBWrapper: All anchors set. Sending all anchors at once and starting positioning!")
            self.send_all_anchors()

    def send_all_anchors(self):
        if self.anchor_count == 0 or not all(self.anchors):
            print("UWBWrapper: Anchors incomplete.")
            return

        def format_coord(val):
            return str(int(val)) if val == int(val) else f"{val:.2f}".rstrip('0').rstrip('.')

        parts = [str(self.anchor_count)]
        for i, (x, y, z) in enumerate(self.anchors):
            parts.append(f"{i+1},{format_coord(x)},{format_coord(y)},{format_coord(z)}")
        message = ";".join(parts) + ";\n"

        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.reset_input_buffer()
                self.serial_port.write(message.encode())
                self.serial_port.flush()
                print(f"UWBWrapper: Sent all anchors: {message.strip()}")

                # 等待確認訊息改用讀取方式來處理破碎資料
                timeout = 5
                start = time.time()
                buffer = ""
                while time.time() - start < timeout:
                    if self.serial_port.in_waiting:
                        data = self.serial_port.read(self.serial_port.in_waiting).decode(errors='ignore')
                        buffer += data
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            print(f"[UWBWrapper] Callback or feedback: {line}")
                            if self.fuzzy_match("Delivery Success", line) or self.fuzzy_match("Start Positioning!", line):
                                print("UWBWrapper: Anchor delivery successful.")
                                self.anchors_sent = True
                                return
                            elif self.fuzzy_match("Delivery Fail", line):
                                print("UWBWrapper: Anchor delivery failed.")
                                return
                            else:
                                print(f"[UWBWrapper] Unrecognized line: {line}")
                print("UWBWrapper: No callback message received.")
            except Exception as e:
                print(f"UWBWrapper: Failed to send all anchors: {e}")

    def start_positioning(self):
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.write(b"les\r\n")
                print("UWBWrapper: Sent start positioning command (les)")
            except Exception as e:
                print(f"UWBWrapper: Failed to send start command: {e}")

    def reset_anchors(self):
        self.anchor_count = 0
        self.anchors = []
        self.latest_position = (0.0, 0.0, 0.0)
        self.anchors_sent = False
        print("UWBWrapper: Reset anchors and count.")

        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.reset_input_buffer()
                self.serial_port.write(b"reset\n")
                self.serial_port.flush()
                print("UWBWrapper: Sent 'reset' command.")
            except Exception as e:
                print(f"UWBWrapper: Failed to reset anchors: {e}")

