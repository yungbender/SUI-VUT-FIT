from collections import defaultdict


class GameSummary:
    def __init__(self):
        self.winner = None
        self.nb_battles = 0
        self.eliminations = []

    def set_winner(self, winner):
        if winner is None:
            self.winner = '#None'
        else:
            self.winner = winner

    def add_battle(self):
        self.nb_battles += 1

    def add_elimination(self, eliminated, battles):
        self.eliminations.append((eliminated, battles))

    def __repr__(self):
        winner_str = 'Winner: {}\n'.format(self.winner)
        nb_battles_str = 'Battles total: {}\n'.format(self.nb_battles)
        total_str = winner_str + nb_battles_str

        for elimination in self.eliminations:
            total_str += 'After {} battles eliminated {}\n'.format(elimination[1], elimination[0])

        return total_str

    def participants(self):
        return [elim[0] for elim in self.eliminations] + [self.winner]

    @classmethod
    def from_repr(cls, str_repr):
        lines = str_repr.split('\n')

        winner = lines[0].split(maxsplit=1)[1]
        nb_battles = int(lines[1].split()[2])

        eliminations = []
        for line in lines[2:]:
            if line == '':
                break
            fields = line.split(maxsplit=4)
            eliminations.append((fields[-1], int(fields[1])))

        summary = cls()
        summary.set_winner(winner)
        summary.nb_battles = nb_battles
        summary.eliminations = eliminations
        return summary


def get_win_rates(summaries, nb_players):
    nb_wins = defaultdict(int)

    for summary in summaries:
        nb_wins[summary.winner] += 1

    return dict(nb_wins)
