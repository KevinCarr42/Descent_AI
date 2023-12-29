# Rules
* Doors are locked until all monsters are defeated in the current area
* If obstacles block entire path, they can be destroyed with an attack action

### Notes
* Enemies in this AI version just use basic hero attacks and ignore overlord powers
* Enemies are able to battle, run, guard, etc., depending on their AI description / role

### Target / Default Target
* First check for ideal target within range = engaged
* If none, DEFAULT TARGET: Most damaged > lowest total HP > lowest defense > heros choice
* repeat with more distant ranges until a target is chosen

### Definitions
* Aim = reroll misses, surges = range
* Battle = make 2 attacks
* Berserk = if the monster has one or more wound tokens on it, it rolls all 5 power dice when attacking
* Dodge = reroll all non-misses on dice with miss icons
* Drain = each HP of drain is treated as a leech attack (-surges, heal monster)
* Fear = attacker must spend 1 surge for every Fear rank or the attack misses
* Flee = move max distance away from characters, maximize distance between monsters
* Form a line = 1st monster ends turn at range to target closest to obstruction/wall, subsequent enemies continue line to block maximum movement, starting at a wall when possible, prioritising the line over their preferred target when targets can be substituted
* Guard = makes an interrupt attack when a hero enters range
* Leech = for each wound lost due to a Leech attack, the target also loses 1 fatigue (or suffers 1 additional wound, ignoring armor, if out of fatigue) and the attacker is healed of 1 wound
* Range = the range at which enemies attacks are optimal (note: enemy orders include ±1 range when checking if "at range")
* Run = move up to 2x speed
* Stealth = like an invisibility potion, roll stealth die to attack, every turn roll power die - surge means no more stealth
* Swarm = a figure with Swarm may roll 1 extra power die for every other friendly figure adjacent to its target. additional enemies attack the initial swarm target (and attacks gain the swarm effect)
* Undying n = Undying effect (roll power die, surge is ressurect) using n dice

### List of Monsters:
* Bane Spider, Beastman, Blood Ape, Chaos Beast, Dark Priest, Deep Elf, Demon, Dragon, Ferrox, Giant, Golem, Hell Hound, Ice Wyrm, Kobold, Lava Beetle, Manticore, Medusa, Naga, Ogre, Razorwing, Shade, Skeleton, Sorcerer, Troll, Wendigo