import hexutil
from typing import List


class Area:
    """Game board area
    """
    def __init__(self, name, owner, dice, neighbours, hexes):
        """
        Parameters
        ----------
        name : int
        owner : int
        dice : int
        neighbours : list of int
        hexes : list of list of int
            Hex coordinates of for all Area's hexes
        """
        self.name = int(name)
        self.owner_name = int(owner)
        self.dice = int(dice)
        self.neighbours = [int(n) for n in neighbours]
        self.hexes = [[int(i) for i in h] for h in hexes]

    def get_adjacent_areas(self) -> List[int]:
        """Return names of adjacent areas
        """
        return self.neighbours

    def get_dice(self) -> int:
        """Return number of dice in the Area
        """
        return self.dice

    def get_name(self) -> int:
        """Return Area's name
        """
        return self.name

    def get_owner_name(self) -> int:
        """Return Area's owner's name
        """
        return self.owner_name

    def can_attack(self) -> bool:
        """Return True if area has enough dice to attack
        """
        return self.dice >= 2

    def set_dice(self, dice: int) -> None:
        """Set area's dice
        """
        if dice < 1 or dice > 8:
            raise ValueError("Attempted to assign {} dice to Area {}".format(dice, self.name))

        self.dice = dice

    def set_owner(self, name: int) -> None:
        """Set owner name
        """
        self.owner_name = int(name)

    ##############
    # UI METHODS #
    ##############
    def get_hexes(self):
        """Return Hex objects of the Area
        """
        return [hexutil.Hex(h[0], h[1]) for h in self.hexes]
