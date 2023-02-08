from abc import ABCMeta, abstractmethod


class SL(metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def init_capture(source):
        """
        :param source: Source object
        method that initialize capture
        :return: cap
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def start(self):
        """
        method that start threading
        :return: self
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def update(self, source):
        """
        :param source: Source object
        method that update frame
        """
        raise NotImplementedError("Method not implemented")

    @property
    @abstractmethod
    def streams(self):
        """
        method that create ids_list, images_list, images_to_detect_list
        :return: ids_list, images_list, images_to_detect_list
        """
        raise NotImplementedError("Method not implemented")
