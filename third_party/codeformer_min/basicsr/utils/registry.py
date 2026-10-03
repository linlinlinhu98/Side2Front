class Registry:
    """最小替代: 官方 basicsr.utils.registry.Registry (只实现 register)"""

    def __init__(self, name):
        self._name = name
        self._obj_map = {}

    def register(self, name=None):
        def deco(func_or_class):
            key = name or func_or_class.__name__
            self._obj_map[key] = func_or_class
            return func_or_class
        return deco

    def get(self, name):
        return self._obj_map.get(name)


ARCH_REGISTRY = Registry('arch')
