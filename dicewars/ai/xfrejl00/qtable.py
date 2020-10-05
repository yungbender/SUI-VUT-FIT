import os
import pickle
import collections


class QTableExc(Exception):
    pass


class QTable(collections.UserDict):
    def __init__(self, states_count=None, action_count=None, qvalue_check=False):
        self.states_count = states_count
        self.action_count = action_count
        self.qvalue_check = qvalue_check

        super().__init__(self)

    def __getitem__(self, key: tuple):
        if not isinstance(key, tuple):
            raise QTableExc("Given key is not type tuple.")
        elif len(key) != 2:
            raise QTableExc("Accepting only keys in value ((state), (action))")

        # Key[0] is state, Key[1] is action
        if self.states_count is not None and len(key[0]) != self.states_count:
            raise QTableExc("Given state in key does not have given number of states.")

        if self.action_count is not None and len(key[1]) != self.action_count:
            raise QTableExc("Given action in key does not have given number of states.")

        return super().__getitem__(key)

    def __setitem__(self, key: tuple, value: int):
        if not isinstance(key, tuple):
            raise QTableExc("Given key is not type tuple.")
        elif len(key) != 2:
            raise QTableExc("Accepting only keys in value ((state), (action))")

        # Key[0] is state, Key[1] is action
        if self.states_count is not None and len(key[0]) != self.states_count:
            raise QTableExc("Given state in key does not have given number of states.")

        if self.action_count is not None and len(key[1]) != self.action_count:
            raise QTableExc("Given action in key does not have given number of states.")

        if self.qvalue_check and not isinstance(value, int):
            raise QTableExc("Given value is not integer.")

        return super().__setitem__(key, value)

    def __setstate__(self, state):
        self.__dict__ = state

    def save(self, where: str):
        with open(where, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(where: str):
        with open(where, "rb") as file:
            return pickle.load(file)
