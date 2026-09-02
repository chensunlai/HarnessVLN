from domain.modules import Module


class GoalDriver(Module):
    def run(self) -> None:
        first = self.context.register.call(
            self.context.name, "env.goal.finish", {"status": "completed"}
        )
        second = self.context.register.call(
            self.context.name, "env.goal.finish", {"status": "completed"}
        )
        self.context.output.write_json("goals.json", [first, second])
