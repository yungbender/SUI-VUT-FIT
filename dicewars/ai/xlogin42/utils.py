def attacker_advantage(attacker, defender):
    return attacker.get_dice() - defender.get_dice()


def best_sdc_attack(attacks):
    scored_attacks = [(source, target, attacker_advantage(source, target)) for source, target in attacks]
    scored_attacks.sort(key=lambda attack: attack[2], reverse=True)

    return scored_attacks[0]


def is_acceptable_sdc_attack(attack):
    source, target, advantage = attack
    if advantage > 0 or source.get_dice() == 8:
        return True
    else:
        return False
