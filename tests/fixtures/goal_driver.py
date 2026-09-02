from domain.modules import Module


class GoalDriver(Module):
    def run(self) -> None:
        initial = self.context.register.call(self.context.name, "env.observe")
        first = self.context.register.call(
            self.context.name, "env.goal.finish", {"status": "completed"}
        )
        next_observation = self.context.register.call(self.context.name, "env.observe")
        second = self.context.register.call(
            self.context.name, "env.goal.finish", {"status": "completed"}
        )
        self.context.output.write_json(
            "goals.json",
            {
                "initial": initial,
                "first_finish": first,
                "next": next_observation,
                "second_finish": second,
            },
        )
