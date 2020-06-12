import ray
@ray.remote
class PatientActorNursery:
    """Initialize and store all actor handles. Taken from Ray serve Nursery.

    Note:
        This actor is necessary because ray will destory actors when the
        original actor handle goes out of scope (when driver exit). Therefore
        we need to initialize and store actor handles in a seperate actor.
    """

    def __init__(self):
        # Dict: Actor handles -> tag
        self.actor_handles = dict()

    def start_actor(self, actor_cls, tag, init_args=(), init_kwargs={}):
        """Start an actor and add it to the nursery"""
        handle = actor_cls.remote(*init_args, **init_kwargs)
        self.actor_handles[handle] = tag
        return [handle]

    def get_all_handles(self):
        return {tag: handle for handle, tag in self.actor_handles.items()}

    def get_handle(self, actor_tag):
        return [self.get_all_handles()[actor_tag]]

    def remove_handle(self, actor_tag):
        [handle] = self.get_handle(actor_tag)
        self.actor_handles.pop(handle)
        del handle
