# Author: ray
# Date: 2/10/24
# Description:
from typing import Any


class SetUpStrategy:
    def set_up(self) -> Any:
        """
        This function set up for the training process.
        Including:
            - Get raw data of news and stock
            - Preprocess the data

        Since this is once for life function, we don't need to consider the efficiency of this function that much.

        Returns:
            - Depend on the scale of dataset, this function could store the preprocessed data in local files or return
        """
        raise NotImplementedError("set up method must be implemented")
