from typing import Callable,Any


class Pipeline(Callable):

    def __init__(self, transfroms : list[Callable]) -> None:
        self.transfroms = transfroms

    def __call__(self, x : Any) -> Any:

        y = x
        
        for transfrom in self.transfroms:
            y = transfrom(y)

        return y