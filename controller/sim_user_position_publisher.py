"""Minimal ROS2 publisher for simulated user position.

Publishes geometry_msgs/msg/PointStamped on /sim/user_position by default.
This is intended as a lightweight data source for SimStateProvider in PX4_SIM mode.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Publish simulated user position to ROS2 topic")
    parser.add_argument("--topic", default="/sim/user_position", help="ROS2 topic name")
    parser.add_argument("--x", type=float, default=8.0, help="User X position")
    parser.add_argument("--y", type=float, default=8.0, help="User Y position")
    parser.add_argument("--z", type=float, default=0.0, help="User Z position")
    parser.add_argument("--rate", type=float, default=10.0, help="Publish rate in Hz")
    return parser.parse_args()


def main():
    args = parse_args()

    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PointStamped

    class SimUserPositionPublisher(Node):
        def __init__(self):
            super().__init__("sim_user_position_publisher")
            self._publisher = self.create_publisher(PointStamped, args.topic, 10)
            self._x = float(args.x)
            self._y = float(args.y)
            self._z = float(args.z)
            period = 1.0 / max(0.1, float(args.rate))
            self._timer = self.create_timer(period, self._publish)
            self.get_logger().info(
                f"Publishing PointStamped to {args.topic} at {1.0/period:.2f} Hz: ({self._x}, {self._y}, {self._z})"
            )

        def _publish(self):
            msg = PointStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "map"
            msg.point.x = self._x
            msg.point.y = self._y
            msg.point.z = self._z
            self._publisher.publish(msg)

    rclpy.init(args=None)
    node = SimUserPositionPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
