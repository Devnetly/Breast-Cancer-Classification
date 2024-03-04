from typing import Any

class LabelMapper:
    """
        A class that maps the labels of a dataset.

        - Usage Examples :

        ```Python
        dataset = datasets.ImageFolder(
            root='./mnist_png/testing',
            target_transform=LabelMapper({
                0:0,
                1:1,
                2:0,
                ...,
                9:1
            })
        )
        ```

        - In the example above we mapped the labels to a boolean indicating wether the 
        number is odd or even.
    """

    def __init__(self, mapper : dict) -> None:
        """
            The constructor of the `LabelMapper` class.

            Arguments:
            - mapper : a dictionary with keys as the labels and the values as the label 
            to be mapped.

            Returns:
            - None.
        """
        self.mapper = mapper

    def __call__(self, y : Any) -> Any:
        """
            Mapps a label.

            Arguments : 
            - y : the label to map.

            Returns:
            - The new label.
        """

        label =  self.mapper.get(y)

        if label is not None:
            return label
        
        return y