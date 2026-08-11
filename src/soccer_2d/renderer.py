import numpy as np


class SoccerRenderer:
    def __init__(self, x_min, y_min, x_max, y_max, agent_radius, ball_radius, render_fps):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.agent_radius = agent_radius
        self.ball_radius = ball_radius
        self.collision_distance = agent_radius + ball_radius
        self.render_fps = render_fps
        self.figure = None
        self.axes = None

    def render(
        self,
        render_mode,
        agent_pos,
        ball_pos,
        goal_pos,
        goal_width,
        steps,
        reward,
        touched_ball,
        terminated,
    ):
        import matplotlib.pyplot as plt

        if self.figure is None:
            self.figure, self.axes = plt.subplots(figsize=(6, 6))

        axes = self.axes
        axes.clear()
        axes.set_xlim(self.x_min, self.x_max)
        axes.set_ylim(self.y_min, self.y_max)
        axes.set_aspect("equal")
        axes.set_xlabel("x")
        axes.set_ylabel("y")
        axes.add_patch(
            plt.Rectangle(
                (self.x_min, self.y_min),
                self.x_max - self.x_min,
                self.y_max - self.y_min,
                fill=False,
                edgecolor="black",
                linewidth=2,
            )
        )

        goal_start = goal_pos.x - goal_width / 2
        goal_end = goal_pos.x + goal_width / 2
        axes.plot(
            [goal_start, goal_end],
            [goal_pos.y, goal_pos.y],
            color="forestgreen",
            linewidth=6,
            solid_capstyle="butt",
        )
        axes.add_patch(
            plt.Circle(
                (ball_pos.x, ball_pos.y),
                self.collision_distance,
                fill=False,
                edgecolor="gray",
                linestyle="--",
                alpha=0.6,
            )
        )
        axes.add_patch(
            plt.Circle(
                (agent_pos.x, agent_pos.y),
                self.agent_radius,
                color="royalblue",
            )
        )
        axes.add_patch(
            plt.Circle(
                (ball_pos.x, ball_pos.y),
                self.ball_radius,
                color="darkorange",
            )
        )

        if touched_ball:
            axes.annotate(
                "",
                xy=(ball_pos.x, ball_pos.y),
                xytext=(agent_pos.x, agent_pos.y),
                arrowprops={"arrowstyle": "->", "color": "crimson"},
            )

        status = "goal" if terminated else "running"
        axes.set_title(f"Step {steps} | reward {reward:+.2f} | {status}")

        if render_mode == "human":
            plt.show(block=False)
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
            plt.pause(1 / self.render_fps)
            return None

        self.figure.canvas.draw()
        image = np.asarray(self.figure.canvas.buffer_rgba())
        return image[:, :, :3].copy()

    def close(self):
        if self.figure is not None:
            import matplotlib.pyplot as plt

            plt.close(self.figure)
            self.figure = None
            self.axes = None
