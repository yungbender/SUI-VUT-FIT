import os
import collections
import shelve

class QTableExc(Exception):
    pass


class QTable(collections.UserDict):
    def __init__(self, states_count=None, action_count=None, qvalue_check=False):
        self.states_count = states_count
        self.action_count = action_count
        self.qvalue_check = qvalue_check
        self.shelve = None

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

    def save(self, where: str, deepcopy=False):
        if deepcopy:
            with shelve.open(where, "c") as sh:
                sh["states_count"] = self.states_count
                sh["action_count"] = self.action_count
                sh["qvalue_check"] = self.qvalue_check
                sh["data"] = self.__dict__["data"]
        else:
            if not self.shelve:
                self.shelve = shelve.open(where, "c", writeback=True)
                self.shelve["states_count"] = self.states_count
                self.shelve["action_count"] = self.action_count
                self.shelve["qvalue_check"] = self.qvalue_check

            self.shelve["data"] = self.__dict__["data"]
            self.shelve.sync()

    @staticmethod
    def load(where: str):
        sh = shelve.open(where, "c", writeback=True)
        qtable = QTable()
        if "data" in sh:
            qtable.states_count = sh["states_count"]
            qtable.action_count = sh["action_count"]
            qtable.qvalue_check = sh["qvalue_check"]
            qtable.__dict__["data"] = sh["data"]
        qtable.shelve = sh
        return qtable

    def close(self):
        if self.shelve:
            self.shelve.close()
            self.shelve = None

    def __del__(self):
        if self.shelve:
            self.close()
