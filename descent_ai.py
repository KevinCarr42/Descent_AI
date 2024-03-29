# imported modules
import random
import pandas as pd
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from IPython.display import display, clear_output

# CONSTANTS

# file paths
SAVED_MAPS_FOLDER = 'saved_maps'
DATA_FOLDER = 'data'

# imported data
MONSTER_DF = pd.read_csv(os.path.join(DATA_FOLDER, 'monster_import.csv')).fillna(False)
MONSTER_DF['rank'] = pd.Categorical(MONSTER_DF['rank'] , categories=['minion', 'master', 'boss'], ordered=True)
MAP_TILE_DF = pd.read_csv(os.path.join(DATA_FOLDER, 'tile_import.csv'))
TERRAIN_DF = pd.read_csv(os.path.join(DATA_FOLDER, 'obstacle_import.csv'))
COLOUR_DF = pd.read_csv(os.path.join(DATA_FOLDER, 'colour_import.csv'))
ARCHETYPES = pd.read_csv(os.path.join(DATA_FOLDER, 'archetype_import.csv'))

# colours
def get_object_colour_id(name_of_colour):
    try:
        colour_id = COLOUR_DF.loc[COLOUR_DF['object'] == name_of_colour, 'id'].iloc[0]
    except:  # all exceptions need to be labelled as 99 for error checking by inspection
        colour_id = 99  # anything not in the 
    return colour_id

class Treasures:
    ALL_AVAILABLE_TREASURE_MARKERS = {
        'healing_potion': 9, 
        'vitality_potion': 9, 
        'money_marker': 29,
        'treasure_chest': 8,  # note: treasure chests should automatically match to area level
        'invulnerability_potion': 9,
        'power_potion': 9,
        'invisibility_potion': 9,
        'relic_marker': 11,
    }

    def __init__(self, n_encounters=4, n_treasure_per_encounter=3, always_chest=True, fancy_chance=0.2, expansions=False):
        if expansions:
            self.available_treasures = self.ALL_AVAILABLE_TREASURE_MARKERS.copy()
        else:
            self.available_treasures = {k:v for k,v in list(self.ALL_AVAILABLE_TREASURE_MARKERS.items())[:4]}
        self.encounters = self.generate_encounters(n_encounters, n_treasure_per_encounter, always_chest, fancy_chance)
        
    def generate_encounters(self, n_encounters, n_treasure_per_encounter, always_chest, fancy_chance):
        encounter_dict = dict()
        for i in range(n_encounters):
            encounter_dict[i+1] = ['treasure_chest'] if always_chest else list()
            self.available_treasures['treasure_chest'] -= 1 if always_chest else 0

        n_total = n_treasure_per_encounter * n_encounters
        n_total -= n_encounters if always_chest else 0
        
        for _ in range(n_total):
            all_treasures = [x for x in self.available_treasures.keys() if self.available_treasures[x] > 0]
            all_treasures = all_treasures[:3] * int(1/fancy_chance) + all_treasures[3:]
            treasure = np.random.choice(all_treasures)
            encounter_dict[np.random.choice(list(encounter_dict.keys()))] += [treasure]
            self.available_treasures[treasure] -= 1

        return encounter_dict


class Monsters:
    # constants
    MIN_POWER_PER_ENCOUNTER = 5
    
    def __init__(self, n_monsters=5, n_encounters=4, boss=True, sorted_battles=True, use_all_minis=False):
        self._MINIONS = sorted(list({x.replace('Master', '').strip() for x in MONSTER_DF[MONSTER_DF.minion].name}))
        self._NOT_MINIONS = sorted(list({x.replace('Master', '').strip() for x in MONSTER_DF[~MONSTER_DF.minion].name}))
        self._BOSSES = sorted(list({x.replace('Master', '').strip() for x in MONSTER_DF[MONSTER_DF.boss].name}))
        self.quest_monsters = None
        self.quest_boss = None
        
        # choose monsters
        self.choose_monsters(n_monsters)
        self.choose_boss(boss)
        
        # summarise relevant data
        self._DATA = MONSTER_DF[MONSTER_DF.name.str.contains('|'.join(self.quest_monsters))]
        self._DATA.loc[:, 'boss'] = False
        if boss:
            boss_df = MONSTER_DF[MONSTER_DF.name.str.contains(self.quest_boss) & MONSTER_DF.boss].copy()
            boss_df['rank'] = 'boss'
            self._DATA = pd.concat([self._DATA, boss_df])
            self._DATA.iloc[-1, 0] = self._DATA.iloc[-1, 0].replace('Master', 'Boss')
        self._DATA = self._DATA.sort_values('rank', ascending=False).reset_index(drop=True)
        n_minions = self._DATA.value_counts('rank')['minion']
        self._DATA.index = [x + 20 - n_minions if x > n_minions else x + 10 for x in self._DATA.index]
        self._DATA = self._DATA
        
        # use summarised data to create list of all monster tokens
        self.minis = list()
        self.used_minis = list()
        self.choose_minis(use_all_minis, n_encounters)
            
        # calculate total power of all minis combined (not including boss, if applicable)
        self._total_power = self.calc_total_power(self.minis[:-1])
        
        # create a list of encounters
        self.encounters = self.generate_encounters(n_encounters, sorted_battles)
        
    def choose_monsters(self, n_monsters):
        n_minions = np.random.randint(1, n_monsters)
        minions = np.random.choice(self._MINIONS, n_minions, replace=False)
        not_minions = np.random.choice(self._NOT_MINIONS, n_monsters-n_minions, replace=False)
        self.quest_monsters = sorted(list(set(minions) | set(not_minions)))
      
    def choose_boss(self, boss):
        if boss:
            self.quest_boss = np.random.choice(self._BOSSES)

    def set_boss_power(self):
        """set boss power to magic number"""
        self._DATA.loc[self._DATA.name.str.contains('boss', case=False), 'power'] = 20
            
    def get_power(self, name_of_monster):
        return self._DATA.loc[self._DATA.name == name_of_monster, 'power'].iloc[0]
    
    def calc_total_power(self, list_of_monsters):
        self.set_boss_power()
        total_power = 0
        for monster in list_of_monsters:
            total_power += self.get_power(monster)
        return total_power
            
    def choose_minis(self, use_all_minis, n_encounters):
        for monster in self.quest_monsters:
            n_normal, n_master = self._DATA.loc[self._DATA.name.str.contains(monster) & ~self._DATA.name.str.contains('Boss'), 'quantity']
            for _ in range(n_normal):
                self.minis.append(monster)
            for _ in range(n_master):
                self.minis.append(monster + ' Master')
        if not use_all_minis:
            n_iterations = 5  # MAGIC NUMBER: more iterations favours larger number of utilised minis
            n_enemies = np.random.randint(n_encounters, len(self.minis)+1, n_iterations).max()
            self.minis = sorted(np.random.choice(self.minis, n_enemies, replace=False))
        if self.quest_boss:
            if self.quest_boss + ' Master' in self.quest_monsters:
                self.minis.remove(self.quest_boss + ' Master')
            self.minis.append(self.quest_boss + ' Boss')
            
    def generate_encounters(self, n_encounters, sorted_battles):
        # confirm min enemies per encounter is possible
        if self.calc_total_power(self.minis[:-1]) < self.MIN_POWER_PER_ENCOUNTER*n_encounters:
            self.choose_minis(True, n_encounters)  # use all minis
        if self.calc_total_power(self.minis[:-1]) < self.MIN_POWER_PER_ENCOUNTER*n_encounters:
            raise TheseMonstersAreNotPowerfulEnoughToMakeEnoughEncounters
        
        # create encounters
        enough_enemies_per_encounter = False
        while not enough_enemies_per_encounter:
            enough_enemies_per_encounter = True
            encounter_dict = dict()
            for i in range(n_encounters):
                encounter_dict[i+1] = list()

            for mini in self.minis[:-1]:
                encounter_dict[np.random.randint(1, n_encounters+1)] += [mini]
            encounter_dict[n_encounters] += [self.minis[-1]]  # add the boss to final encounter
        
            # confirm that all encounters have enough enemies, otherwise try again
            for encounter in encounter_dict.values():
                if self.calc_total_power(encounter) < self.MIN_POWER_PER_ENCOUNTER:
                    enough_enemies_per_encounter = False

        # sort the non-boss items by difficulty
        if sorted_battles:
            difficulty = dict()
            for i in range(1, n_encounters):
                difficulty[i] = self.calc_total_power(encounter_dict[i])

            sorted_dict = encounter_dict.copy()
            for i, k in enumerate(sorted(difficulty, key=difficulty.get, reverse=False)):
                sorted_dict[i+1] = encounter_dict[k]
            
            encounter_dict = sorted_dict

        return encounter_dict
            
    def summarise(self):
        print('\nQuest Monsters\n================')
        for monster in self.quest_monsters:
            print(monster)
        print()
        
        if self.quest_boss:
            print('\nQuest Boss\n================')
            print(self.quest_boss)
            print()

    def show_encounters(self):
        for i, encounter in self.encounters.items():
            print(f'Encounter {i}:\n{", ".join(encounter)}\n')


class Tiles:
    def __init__(self, obj, is_map=False):
        # empty default tileset - fills up as tiles are used
        self.tiles = dict()
        self.unused_tiles = dict()
        
        # generate all tiles
        if isinstance(obj, pd.DataFrame):
            if is_map:
                self.create_map_tiles(obj)
            else:
                self.create_tiles(obj)
        elif isinstance(obj, Monsters):
            self.create_monster_tiles(obj)
        else:
            raise NotImplementedError
        
    def create_tiles(self, dataframe):
        fill_value = 1
        for _, row in dataframe.iterrows():
            if 'colour' in dataframe.columns:
                fill_value = get_object_colour_id(row['colour'])
            for j in range(row['quantity']):
                name = row['id'] + '_' + str(j + 1)
                tile = pd.DataFrame(np.full(shape=(row['height'], row['width'], ), fill_value=fill_value))
                self.unused_tiles[name] = tile

    def create_map_tiles(self, dataframe):
        for _, row in dataframe.iterrows():
            for j in range(row['quantity']):
                name = row['id'] + '_' + str(j + 1)
                tile = pd.DataFrame(np.ones((row['height'], row['width'])))
                # blanks
                if pd.notna(row['blanks']):
                    for blank in eval(row['blanks']):
                        tile.iloc[blank] = None
                # spawn points
                if pd.notna(row['spawn']):
                    for x, y in eval(row['spawn']):
                        tile.iloc[x, y] = 0.5
                self.unused_tiles[name] = tile
                
    def create_monster_tiles(self, obj):
        # these should automatically have rows consistent with plotting colour ranges
        dataframe = obj._DATA.copy()
        for idx, row in dataframe.iterrows():
            for j in range(row['quantity']):
                name = row['name'] + ' ' + str(j + 1)
                tile = pd.DataFrame(np.ones((row['height'], row['width']))) * idx
                self.unused_tiles[name] = tile        
        
    def rotate_tile(self, name_of_tile):
        if name_of_tile[0].lower() == 's':
            return
        t = self.unused_tiles[name_of_tile]
        t = t.sort_index(ascending=False).T.reset_index(drop=True)
        self.unused_tiles[name_of_tile] = t.rename({x:i for i, x in enumerate(t.columns)}, axis=1)
        
    def show_tile(self, name_of_tile, unused=True):
        if unused:
            display(self.unused_tiles[name_of_tile])
        else:
            display(self.tiles[name_of_tile])
    
    def use_tile(self, name_of_tile):
        self.tiles[name_of_tile] = self.unused_tiles.pop(name_of_tile)


class DescentScenario:
    # cls constants
    DEBUGGING = False
    CROP_MAP = True
    BOSS_POWER_LIST = ['double HP', 'double attack dice', 'resurrect unless miss on 3 red']
    
    def __init__(self, n_monsters=5, n_encounters=4, n_treasure_per_encounter=3, boss=True, sorted_battles=True, use_all_minis=False, 
                 always_chest=True, unrevealed_areas=True, MAX_H=50, MAX_W=40, min_area_size=30, max_area_size=80):
        # obj constants
        self.MAX_H = MAX_H
        self.MAX_W = MAX_W
        self.MAP_TEMPLATE = pd.DataFrame(np.full(shape=(MAX_H, MAX_W, ), fill_value=np.nan))
        self.N_ENCOUNTERS = n_encounters
        self.MIN_AREA_SIZE = min_area_size
        self.MAX_AREA_SIZE = max_area_size
        self._n_monsters = n_monsters
        self._n_treasure_per_encounter = n_treasure_per_encounter
        self._boss = boss
        self._sorted_battles = sorted_battles
        self._use_all_minis = use_all_minis
        self._n_treasure_per_encounter = n_treasure_per_encounter
        self._always_chest = always_chest

        # create objects
        self.monsters_obj = Monsters(self._n_monsters, self.N_ENCOUNTERS, self._boss, self._sorted_battles, self._use_all_minis)
        self.treasures_obj = Treasures(self.N_ENCOUNTERS, self._n_treasure_per_encounter, self._always_chest)
        self.map_tiles_obj = Tiles(MAP_TILE_DF, True)
        self.monster_tiles_obj = Tiles(self.monsters_obj)
        self.terrain_tiles_obj = Tiles(TERRAIN_DF)
        self.ai_overlord = AIOverlord(self.monsters_obj.quest_monsters, self.monsters_obj.quest_boss)

        # tracking all map object attributes
        self.encounters = {
            'map_tiles': {k+1:list() for k in range(self.N_ENCOUNTERS)},
            'monsters': {i+1:x for i, x in enumerate(self.monsters_obj.encounters.values())},
            'treasures': {i+1:x for i, x in enumerate(self.treasures_obj.encounters.values())},
            'terrain': {k+1:list() for k in range(self.N_ENCOUNTERS)},
        }
        
        # initialise variables
        self._current_area = 1  # TODO: this is now just used to generate different areas, should be underscored
        self._current_area_size = 0
        self.choose_area_size()
        self.area_dict = {n+1:list() for n in range(n_encounters)}
        self.label_dict = self.ai_overlord.label_dict
        self._dungeon_terrain_type = None
        
        # layers for plotting
        self.grid_map = self.MAP_TEMPLATE.copy()
        self.area_map = self.MAP_TEMPLATE.copy()
        self.terrain_map = self.MAP_TEMPLATE.copy()
        self.monster_map = self.MAP_TEMPLATE.copy()
        self.treasure_map = self.MAP_TEMPLATE.copy()
        self.label_map = self.MAP_TEMPLATE.copy()
        
        # set up the dungeon
        self.create_dungeon_entrance()

    def save(self, file_name=None):
        folder = SAVED_MAPS_FOLDER
        if not file_name:
            file_name = f'{datetime.now().strftime("%Y%m%d_%H%M")}.pickle'
        with open(os.path.join(folder, file_name), 'wb') as f:
            pickle.dump(self, f)
        print(f'{os.path.join(folder, file_name)} saved.')

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def add_tile_to_encounter_list(self, name_of_tile, encounter_str):
        self.encounters[encounter_str][self._current_area].append(name_of_tile)
    
    def place_tile(self, name_of_tile, x, y, tiles_obj, map_obj):
        tile = tiles_obj.unused_tiles[name_of_tile]
        w, h = tile.shape
        map_obj.iloc[y:y+w, x:x+h] = tile
        tiles_obj.use_tile(name_of_tile)
    
    def add_map_tile_to_current_area(self, name_of_tile):
        self.area_dict[self._current_area].append(name_of_tile)

    def _fill_area_map(self):
        mask = self.grid_map.notnull() & self.area_map.isnull()
        self.area_map[mask] = self._current_area
    
    def place_map_tile(self, name_of_tile, x, y):
        tile_obj = self.map_tiles_obj
        self.add_map_tile_to_current_area(name_of_tile)
        self.add_tile_to_encounter_list(name_of_tile, 'map_tiles')
        self.place_tile(name_of_tile, x, y, tile_obj, self.grid_map)
        self._fill_area_map()

    def cleaned_monster_name(self, full_monster_name, clean_boss=False):
        if full_monster_name[-1].isdigit():
            full_monster_name = ' '.join(full_monster_name.split(' ')[:-1])
        if clean_boss:
            return full_monster_name.replace('Master', '').replace('Boss', '').strip()
        return full_monster_name.replace('Master', '').strip()
            
    def place_monster_tile(self, name_of_tile, x, y):
        tile_obj = self.monster_tiles_obj
        self.place_tile(name_of_tile, x, y, tile_obj, self.monster_map)
        self.label_map.iloc[y, x] = self.label_dict[self.cleaned_monster_name(name_of_tile)]
    
    def place_terrain_tile(self, name_of_tile, x, y):
        tile_obj = self.terrain_tiles_obj
        self.place_tile(name_of_tile, x, y, tile_obj, self.terrain_map)
    
    def place_treasure(self, name_of_item, x, y):
        n_mapping = COLOUR_DF.loc[COLOUR_DF.object==name_of_item.split('_')[0].upper(), 'id'].iloc[0]
        self.treasure_map.iloc[y, x] = n_mapping
    
    def create_dungeon_entrance(self, only_terrain=False):
        """hardcoded spawn location / configuration - consider revising"""
        OFFSET_FROM_BOTTOM = 1
        START_X, START_Y = self.MAX_W // 2, self.MAX_H - OFFSET_FROM_BOTTOM
        if not only_terrain:
            self.place_map_tile('Cap_1', START_X, START_Y)
            self.place_map_tile('H2_1', START_X, START_Y-2)
            self.grid_map.iloc[START_Y-1:START_Y+1, START_X:START_X+2] = 1
        self.place_terrain_tile('dungeon_entrance_1', START_X, START_Y)
        self.place_terrain_tile('dungeon_entrance_2', START_X+1, START_Y)
        self.place_terrain_tile('dungeon_entrance_3', START_X, START_Y-1)
        self.place_terrain_tile('dungeon_entrance_4', START_X+1, START_Y-1)
        self.place_terrain_tile('dungeon_entrance_5', START_X, START_Y-2)
        self.place_terrain_tile('dungeon_entrance_6', START_X+1, START_Y-2)
    
    @property
    def spawn_points(self):
        filtered = self.grid_map.applymap(lambda x: x == 0.5).stack()
        return [(x, y) for y,x in filtered[filtered].index.tolist()]
    
    @property
    def size(self):
        return self.grid_map.count().sum()

    @property
    def current_area_size(self):
        return (self.area_map == self._current_area).sum().sum()
    
    @property
    def tiles(self):
        return self.map_tiles_obj.tiles

    @property
    def unused_tiles(self):
        return self.map_tiles_obj.unused_tiles

    @property
    def combined_map(self):
        combined_map = self.grid_map.copy()
        for layer in [self.terrain_map, self.treasure_map, self.monster_map]:
            combined_map = layer.combine_first(combined_map)
        if self.CROP_MAP:
            combined_map = combined_map.loc[combined_map.notnull().any(axis=1), combined_map.notnull().any(axis=0)]
        return combined_map

    def show_areas(self, crop=True):
        area_map = self.area_map.copy()
        if crop:
            area_map = area_map.loc[area_map.notnull().any(axis=1), area_map.notnull().any(axis=0)]
        return area_map.fillna('')
    
    def show_monsters(self, crop=False):
        monster_map = self.monster_map.copy()
        if crop:
            monster_map = monster_map.loc[monster_map.notnull().any(axis=1), monster_map.notnull().any(axis=0)]
        return monster_map.fillna('')

    @property
    def tiles_remaining(self):
        """not including caps"""
        return len([x for x in self.unused_tiles if x[:3] != 'Cap'])

    @property
    def caps_remaining(self):
        return len([x for x in self.unused_tiles if x[:3] == 'Cap'])
    
    @property
    def distinct_spawn_points(self):
        to_cap = self.spawn_points
        for i, coordinates in enumerate(to_cap):
            paired_coordiates = self.get_spawns_pair_coordinates(*coordinates)
            if paired_coordiates in to_cap:
                to_cap[i] = paired_coordiates

        # sorted so that lower/outer spawn points are capped first (greater y value, lower on map)
        return sorted(list(set(to_cap)), key=lambda x: x[1] + abs(x[0]-int(self.MAX_W/2)), reverse=True)

    @property
    def spawn_points_remaining(self):
        return len(self.distinct_spawn_points)

    @property
    def new_tile_mask(self):
        new_tile_mask = self.grid_map.copy()
        for x, y in self.distinct_spawn_points:
            new_tile_mask = self.spawn_square_df(x, y).combine_first(new_tile_mask)
        return new_tile_mask
    
    @property
    def number_of_big_tiles(self):
        return len([x for x in self.unused_tiles if x[0]=='S' or x[0]=='R'])
    
    def ai_summary(self, show_boss=True, monsters_to_display=None):
        return self.ai_overlord.summary(show_boss, monsters_to_display)

    def ai_boss_summary(self):
        return self.ai_overlord.boss_summary()
    
    def print_ai_summary(self):
        print(self.ai_summary())
    
    def print_ai_boss_summary(self):
        print(self.ai_boss_summary())

    @property
    def ai_overlord_attributes(self):
        return self.ai_overlord.ai_overlord_attributes

    # procedural map generation

    def choose_random_tile(self):
        if self.tiles_remaining == 0:
            if self.caps_remaining == 0:
                return None
            return self.get_cap()
        
        # more halls at start of area and when there are less squares rectangles
        n_attempts_without_hallways = len([x for x in self.unused_tiles if x[0]=='S' or x[0]=='R']) * self.current_area_size // self.MAX_AREA_SIZE
        if self._current_area == self.N_ENCOUNTERS:
            n_attempts_without_hallways *= 5  # less halls in boss room
        name_of_tile = np.random.choice([x for x in self.unused_tiles if x[:3] != 'Cap'])
        for _ in range(n_attempts_without_hallways):
            if name_of_tile[0] not in ['H', 'L', 'X', 'T']:
                break
            name_of_tile = np.random.choice([x for x in self.unused_tiles if x[:3] != 'Cap'])
        return name_of_tile
    
    def use_tile(self, name_of_tile):
        self.map_tiles_obj.use_tile(name_of_tile)

    def choose_random_spawn_point(self):
        return self.spawn_points[np.random.randint(0, len(self.spawn_points))]

    def get_spawns_pair_coordinates(self, x, y):
        coordinate = None

        directions = dict(
            up =  self.grid_map.iloc[y-1, x-1:x+2],
            down = self.grid_map.iloc[y+1, x-1:x+2],
            left = self.grid_map.iloc[y-1:y+2, x-1],
            right = self.grid_map.iloc[y-1:y+2, x+1],
        )
        if (directions['up'].iloc[1] == 0.5).any():
            coordinate = (x, y-1)
        if (directions['down'].iloc[1] == 0.5).any():
            coordinate = 'ERROR' if coordinate else (x, y+1)
        if (directions['left'].iloc[1] == 0.5).any():
            coordinate = 'ERROR' if coordinate else (x-1, y)
        if (directions['right'].iloc[1] == 0.5).any():
            coordinate = 'ERROR' if coordinate else (x+1, y)

        return coordinate
    
    def get_spawn_direction(self, x1, y1):
        direction = None
        x2, y2 = self.get_spawns_pair_coordinates(x1, y1)
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        directions = dict(
            up =  self.grid_map.iloc[y_min-1, x_min:x_max+1],
            down = self.grid_map.iloc[y_min+1, x_min:x_max+1],
            left = self.grid_map.iloc[y_min:y_max+1, x_min-1],
            right = self.grid_map.iloc[y_min:y_max+1, x_min+1],
        )

        if y_min == y_max:
            if directions['up'].isnull().all() and (directions['down'].iloc[1] != 0.5).all():
                direction = 'up'
            if directions['down'].isnull().all() and (directions['up'].iloc[1] != 0.5).all():
                direction = 'ERROR' if direction else 'down'
        if x_min == x_max:
            if directions['left'].isnull().all() and (directions['right'].iloc[1] != 0.5).all():
                direction = 'ERROR' if direction else 'left'
            if directions['right'].isnull().all() and (directions['left'].iloc[1] != 0.5).all():
                direction = 'ERROR' if direction else 'right'

        return direction
    
    def get_spawn_square(self, x1, y1, direction):
        if self.DEBUGGING:
            print('get_spawn_square')
            print(f'square start: {x1}, {y1}')
        x3, y3 = x1, y1
        x4, y4 = x2, y2 = self.get_spawns_pair_coordinates(x1, y1)
        if self.DEBUGGING:
            print(f'pair coordicates: {x2}, {y2}')
        if direction == 'up':
            y3 -= 1
            y4 -= 1
        elif direction == 'down':
            y3 += 1
            y4 += 1
        elif direction == 'left':
            x3 -= 1
            x4 -= 1
        elif direction == 'right':
            x3 += 1
            x4 += 1
        return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    
    def spawn_square_df(self, x, y, fill_value=1):
        ss = self.get_spawn_square(x, y, self.get_spawn_direction(x, y))
        return pd.DataFrame(np.full(shape=(2, 2, ), fill_value=fill_value), 
                            columns=sorted(list({x[0] for x in ss})), 
                            index=sorted(list({x[1] for x in ss})))

    def confirm_two_adjacent_values(self, series, value=0.5):
        if (series == value).sum() != 2:
            return False
        indices = series[series == value].index
        return indices[1] == indices[0] + 1

    def confirm_spawn_possible(self, name_of_tile, x, y):
        direction = self.get_spawn_direction(x, y)
        if direction:
            if direction == 'up':
                series = self.map_tiles_obj.unused_tiles[name_of_tile].iloc[-1, :]
            elif direction == 'down':
                series = self.map_tiles_obj.unused_tiles[name_of_tile].iloc[0, :]
            elif direction == 'left':
                series = self.map_tiles_obj.unused_tiles[name_of_tile].iloc[:, -1]
            elif direction == 'right':
                series = self.map_tiles_obj.unused_tiles[name_of_tile].iloc[:, 0]
        else:
            return False
        return self.confirm_two_adjacent_values(series)

    def rotate_tile(self, name_of_tile):
        if name_of_tile in self.map_tiles_obj.unused_tiles.keys():
            self.map_tiles_obj.rotate_tile(name_of_tile)
        elif name_of_tile in self.monster_tiles_obj.unused_tiles.keys():
            self.monster_tiles_obj.rotate_tile(name_of_tile)
        elif name_of_tile in self.terrain_tiles_obj.unused_tiles.keys():
            self.terrain_tiles_obj.rotate_tile(name_of_tile)
        else:
            raise ThatTileIsntInUnusedListForAnyOfTheMaps

    def tile_spawn_offset(self, name_of_tile, x, y):
        x1, y1 = x, y
        direction = self.get_spawn_direction(x, y)
        spawn_square = self.get_spawn_square(x, y, direction)
        x_min_spawn = min(x[0] for x in spawn_square)
        y_max_spawn = min(x[1] for x in spawn_square) # reversed because of inverted y-axis
        
        if self.DEBUGGING:
            print('tile_spawn_offset')
            print(f"'{name_of_tile}', {x}, {y}")
            print(direction)
        
        for _ in range(4):
            if self.DEBUGGING:
                print()
                print('spawn possible?', self.confirm_spawn_possible(name_of_tile, x, y))
                
            tile = self.map_tiles_obj.unused_tiles[name_of_tile]
            h, w = tile.shape
            if self.DEBUGGING:
                display(tile)
            if self.confirm_spawn_possible(name_of_tile, x, y):
                if direction == 'left':
                    x_offset = -w
                    tile_spawn_pt_location = -tile.iloc[:, -1].eq(0.5).idxmax()
                    offset_correction = y_max_spawn - y
                    y_offset = offset_correction + tile_spawn_pt_location
                    if self.DEBUGGING:
                        print('tile point', tile_spawn_pt_location)
                        print('offset correction', offset_correction, y)
                        print('total offset', y_offset)
                    x1, y1 = x + x_offset, y + y_offset
                elif direction == 'right':
                    x_offset = 1
                    tile_spawn_pt_location = -tile.iloc[:, 0].eq(0.5).idxmax()
                    offset_correction = y_max_spawn - y
                    y_offset = offset_correction + tile_spawn_pt_location
                    x1, y1 = x + x_offset, y + y_offset
                    if self.DEBUGGING:
                        print('tile point', tile_spawn_pt_location)
                        print('offset correction', offset_correction, y)
                        print('total offset', y_offset)
                elif direction == 'up':
                    y_offset = -h
                    tile_spawn_pt_location = -tile.iloc[-1, :].eq(0.5).idxmax()
                    offset_correction = x_min_spawn - x
                    x_offset = tile_spawn_pt_location + offset_correction
                    if self.DEBUGGING:
                        print('tile point', tile_spawn_pt_location)
                        print('offset correction', offset_correction, y)
                        print('total offset', y_offset)
                    x1, y1 = x + x_offset, y + y_offset
                elif direction == 'down':
                    y_offset = 1
                    tile_spawn_pt_location = -tile.iloc[0, :].eq(0.5).idxmax()
                    offset_correction = x_min_spawn - x
                    x_offset = tile_spawn_pt_location + offset_correction
                    if self.DEBUGGING:
                        print('tile point', tile_spawn_pt_location)
                        print('offset correction', offset_correction, y)
                        print('total offset', y_offset)
                    x1, y1 = x + x_offset, y + y_offset
                if self.DEBUGGING:
                    print(f'location to spawn: {x1}, {y1}')
                    print()
                return x1, y1
            else:
                if self.DEBUGGING:
                    print('ROTATE!')
                    print()
                self.rotate_tile(name_of_tile)

        return None  # if there are no eligible spawn points, return None

    def confirm_spawn_fits(self, name_of_tile, x_o, y_o, direction, x_s, y_s):
        """_o denotes origin coordinate, _s denotes spawn point coordinate"""
        tile = self.map_tiles_obj.unused_tiles[name_of_tile]
        h, w = tile.shape

        # check if tile fits in grid with 1 cell buffer
        if x_o < 1 or x_o+w > self.MAX_W-1 or y_o < 1 or y_o+h > self.MAX_H-1:
            return False
        
        # check if tile fits in all null areas with 1 cell buffer (will include room to at least cap all spawn points)
        if name_of_tile[:3] == 'Cap':
            # caps don't need buffers
            left, right, up, down = 0, 0, 0, 0
        else:
            left = 0 if direction == 'right' else 1
            right = 0 if direction == 'left' else 1
            up = 0 if direction == 'down' else 1
            down = 0 if direction == 'up' else 1
        if self.DEBUGGING:
            print('confirm_spawn_fits')
            print(f'h: {h}, w: {w}, direction: {direction}')
            print(f'left: {left}, right: {right}, up: {up}, down: {down}')
            print(f'.iloc[{y_o-up}:{y_o+h+down}, {x_o-left}:{x_o+w+right}].isnull().all().all()')
            display(self.grid_map.iloc[y_o-up:y_o+h+down, x_o-left:x_o+w+right])
            print('does it fits? ', self.grid_map.iloc[y_o-up:y_o+h+down, x_o-left:x_o+w+right].isnull().all().all())
            print()

        spawn_mask = self.new_tile_mask
        spawn_mask[self.spawn_square_df(x_s, y_s, True)] = np.nan
        return spawn_mask.iloc[y_o-up:y_o+h+down, x_o-left:x_o+w+right].isnull().all().all()

    def clean_used_spawn_square(self, list_spawn_square_output):
        for x, y in list_spawn_square_output:
            self.grid_map.iloc[y, x] = 9.9 if self.DEBUGGING else 1

    def randomly_rotate_tile(self, name_of_tile):
        n_rotations = np.random.choice(range(4))
        for _ in range(n_rotations):
            self.rotate_tile(name_of_tile)

    def get_cap(self):
        return [x for x in self.unused_tiles if x[:3] == 'Cap'][0]

    def spawn_tile(self, name_of_tile=None, spawn_point=None, show=False):
        # check that there are available spawn points
        if len(self.spawn_points) == 0:
            print('Nowhere to spawn tiles...\n')
            if show:
                display(self.grid_map.fillna(''))
            return

        # properties of the spawn point and it's tile
        x, y = spawn_point if spawn_point else self.choose_random_spawn_point()
        direction = self.get_spawn_direction(x, y)
        spawn_square_output = self.get_spawn_square(x, y, direction)

        # properties of the new tile
        name_of_tile = name_of_tile if name_of_tile else self.choose_random_tile()

        # randomly rotate tile
        self.randomly_rotate_tile(name_of_tile)

        # if the first tile doesn't fit, attempt a few more before giving up and capping
        how_many_attempts = 10  # TODO: maybe make this a parameter, or just leave as a magic number
        for _ in range(how_many_attempts):
            # attempt to place the tile, rotate if required
            for _ in range(4):
                tile_offset = self.tile_spawn_offset(name_of_tile, x, y)
                if tile_offset:
                    x1, y1 = tile_offset
                    if self.confirm_spawn_fits(name_of_tile, x1, y1, direction, x, y):
                        self.place_map_tile(name_of_tile, x1, y1)
                        self.clean_used_spawn_square(spawn_square_output)

                        if show:
                            display(self.grid_map.fillna(''))
                        return
                else:  
                    self.rotate_tile(name_of_tile)  # rotate tile
                    if self.DEBUGGING:
                        print('ERROR: tile cannot be placed. rotating tile.')
            # before capping, try a few more tiles
            name_of_tile = self.choose_random_tile()

        # if no eligible rotations found, cap the end
        if self.caps_remaining <= 0:
            print('BIG ERROR: map cannot be completed. no caps remain.') # TODO: remove this when working, consider capping end
            return

        name_of_tile = self.get_cap()
        for _ in range(2):
            tile_offset = self.tile_spawn_offset(name_of_tile, x, y)
            if tile_offset:
                self.place_map_tile(name_of_tile, tile_offset[0], tile_offset[1])
                self.clean_used_spawn_square(spawn_square_output)

                if show:
                    display(self.grid_map.fillna(''))
                return
            else:  
                self.rotate_tile(name_of_tile)  # rotate tile
                print('BIG ERROR: map cannot be completed... but I am not really sure why...') # TODO: remove this when working, consider capping end

    def cap_if_close_to_edge(self, dist=5):
        if self.spawn_points_remaining <= 1:
            return
        for x, y in self.distinct_spawn_points:
            if (x < dist) or (x > self.MAX_W - dist) or (y < dist) or (y > self.MAX_H - dist):
                self.spawn_tile(self.get_cap(), (x, y))

    def cap_lowest_spawn_points(self, n_to_leave=3):
        if self.spawn_points_remaining <= n_to_leave:
            return
        for x, y in self.distinct_spawn_points[:-n_to_leave]:
            self.spawn_tile(self.get_cap(), (x, y))

    def cap_all_except_n(self, n=1):  # TODO: how did this get messed up? it worked before and I didn't change anything related to it...
        if self.spawn_points_remaining > n:
            xy_to_cap = random.sample(self.distinct_spawn_points, len(self.distinct_spawn_points) - n)
        else:
            return
        for x, y in xy_to_cap:
            self.spawn_tile(self.get_cap(), (x, y))

    def choose_area_size(self):
        self._current_area_size = np.random.randint(self.MIN_AREA_SIZE, self.MAX_AREA_SIZE)
        
    def increment_area_number(self):
        self._current_area += 1
        
    def set_revealed_area(self, n):
        """just in case of accidental reveal, panic undo button"""
        if isinstance(n, int):
            self._current_area = n

    @property
    def _map_unfinishable(self):
        return self.caps_remaining < len(self.distinct_spawn_points)

    def _create_map_loop(self, show=False):
        for _ in range(self.N_ENCOUNTERS):
            if show:
                clear_output(wait=True)
                display(self.show_areas(False))

            while self.current_area_size < self._current_area_size:
                if (self.tiles_remaining == 0 
                    or self.spawn_points_remaining == 0
                    or self._map_unfinishable):
                    if self.DEBUGGING:
                        print("Whelp... That didn't work.")
                    if show:
                        print('exit _create_map_loop... unfinishable?:')
                        display(self.area_map.fillna(''))
                    return
                self.spawn_tile()

            if self._map_unfinishable:
                if show:
                    print('UNFINISHABLE:')
                    display(self.area_map.fillna(''))
                return

            # self.cap_if_close_to_edge()
            self.cap_lowest_spawn_points()

            # choose connection to next area
            n = 1 if self._current_area < self.N_ENCOUNTERS else 0
            self.cap_all_except_n(n)
            if self._current_area < self.N_ENCOUNTERS:
                self.increment_area_number()
                self.choose_area_size()

    def create_map(self, n_attempts=10, show=False):
        for i in range(n_attempts):
            self.__init__()
            if self.DEBUGGING:
                print(f'\nMAP CREATION ATTEMPT # {i+1}')
            self._create_map_loop()
            if self._current_area == self.N_ENCOUNTERS:
                if self.current_area_size >= self.MIN_AREA_SIZE:
                    if self._map_unfinishable:
                        continue
                    if self.DEBUGGING:
                        print('MAP COMPLETED!')
                    if show:
                        clear_output(wait=True)
                        display(self.show_areas())
                    return

    def create_terrain(self, terrain_type=None, use_all=False):
        glyphs = [x for x in list(self.terrain_tiles_obj.unused_tiles.keys()) if x[:5] == 'glyph']
        all_terrain = [x for x in list(self.terrain_tiles_obj.unused_tiles.keys()) if x[:5] != 'glyph']

        terrain_types = ['hot', 'wet', 'dry', 'crazytown']
        if terrain_type and terrain_type in terrain_types:
            self._dungeon_terrain_type = terrain_type
        else:
            self._dungeon_terrain_type = random.choice(terrain_types)

        if self._dungeon_terrain_type == 'hot':
            all_terrain = [x for x in all_terrain if x[:3]!='mud' and x[:5]!='water']
        if self._dungeon_terrain_type == 'wet':
            all_terrain = [x for x in all_terrain if x[:4]!='lava' and x[:3]!='pit']
        if self._dungeon_terrain_type == 'dry':
            all_terrain = [x for x in all_terrain if x[:3]!='mud' and x[:4]!='lava' and x[:5]!='water']

        for i in range(self.N_ENCOUNTERS):  # add 1 glyph to each encounter
            self.encounters['terrain'][i+1] += [glyphs.pop(0)]
        terrain = all_terrain if use_all else list(np.random.choice(all_terrain, np.random.randint(len(all_terrain)), replace=False))
        for obstacle in terrain:
            self.encounters['terrain'][np.random.randint(self.N_ENCOUNTERS)+1] += [obstacle]
        
    def check_placement_legal(self, x, y, tile_df=None):
        if tile_df is None:
            return self.combined_map.loc[y, x] == 1
        h, w = tile_df.shape
        return self.combined_map.loc[y:y+h-1, x:x+w-1].eq(1).sum().sum() == tile_df.size
    
    def get_list_of_coords_in_area(self, n_area):
        coords = [(col, idx) for idx, row in self.area_map.iterrows() for col in self.area_map.columns if row[col] == n_area]
        random.shuffle(coords)
        return coords

    def distribute_monsters(self):
        unused_tiles = list(self.monster_tiles_obj.unused_tiles.keys())
        for n_area in range(1, self.N_ENCOUNTERS+1):
            n_area_monsters = self.encounters['monsters'][n_area].copy()
            for i, monster in enumerate(n_area_monsters):
                for j, tile in enumerate(unused_tiles):
                    if monster == ' '.join(tile.split(' ')[:-1]):
                        n_area_monsters[i] = unused_tiles.pop(j)
                        break
            area_spawn_points = self.get_list_of_coords_in_area(n_area)
            for tile_name in n_area_monsters:
                rectangular = (self.monster_tiles_obj.unused_tiles[tile_name].shape[0] != self.monster_tiles_obj.unused_tiles[tile_name].shape[1])
                for x, y in area_spawn_points:
                    if self.check_placement_legal(x, y, self.monster_tiles_obj.unused_tiles[tile_name]):
                        self.place_monster_tile(tile_name, x, y)
                        area_spawn_points.remove((x, y))
                        break
                    if rectangular:
                        self.rotate_tile(tile_name)
                        if self.check_placement_legal(x, y, self.monster_tiles_obj.unused_tiles[tile_name]):
                            self.place_monster_tile(tile_name, x, y)
                            area_spawn_points.remove((x, y))
                            break

    def distribute_terrain(self):
        for n_area in range(1, self.N_ENCOUNTERS+1):
            n_area_terrain = self.encounters['terrain'][n_area].copy()
            area_spawn_points = self.get_list_of_coords_in_area(n_area)
            for tile_name in n_area_terrain:
                rectangular = (self.terrain_tiles_obj.unused_tiles[tile_name].shape[0] != self.terrain_tiles_obj.unused_tiles[tile_name].shape[1])
                for x, y in area_spawn_points:
                    if self.check_placement_legal(x, y, self.terrain_tiles_obj.unused_tiles[tile_name]):
                        self.place_terrain_tile(tile_name, x, y)
                        area_spawn_points.remove((x, y))
                        break
                    if rectangular:
                        self.rotate_tile(tile_name)
                        if self.check_placement_legal(x, y, self.terrain_tiles_obj.unused_tiles[tile_name]):
                            self.place_terrain_tile(tile_name, x, y)
                            area_spawn_points.remove((x, y))
                            break

    def distribute_treasures(self):
        for n_area in range(1, self.N_ENCOUNTERS+1):
            n_area_treasure = self.encounters['treasures'][n_area].copy()
            area_spawn_points = self.get_list_of_coords_in_area(n_area)
            for treasure in n_area_treasure:
                for x, y in area_spawn_points:
                    if self.check_placement_legal(x, y):
                        self.place_treasure(treasure, x, y)
                        area_spawn_points.remove((x, y))
                        break

    def create_quest(self):
        print('creating map...')
        self.create_map()
        print('distributing monsters...')
        self.distribute_monsters()
        print('distributing terrain...')
        self.create_terrain()
        self.distribute_terrain()
        print('distributing treasures...')
        self.distribute_treasures()
        print('complete!')

    # plotting and display function
        
    def summarise_scenario(self):
        for x in ['map_tiles', 'monsters', 'terrain', 'treasures']:
            print(x.title().replace('_', ' '))
            if self.encounters[x]:
                for k, v in self.encounters[x].items():
                    print(f'Encounter {k}: {v}')
            print()

    def show(self, show_up_to_area_n=None, show_boss=None):
        if show_up_to_area_n:
            combined_map_to_plot = self.combined_map[self.area_map<=show_up_to_area_n]
        else:
            combined_map_to_plot = self.combined_map

        SCALE = 0.5
        x, y = combined_map_to_plot.shape
        label_size = 16
        
        cmap = ListedColormap(COLOUR_DF.colour.to_list())
        norm = BoundaryNorm(
            COLOUR_DF.id.to_list() + [max(COLOUR_DF.id.to_list())+1], 
            cmap.N, 
            clip=True
        )
        
        plt.figure(figsize=(SCALE*y, SCALE*x))
        plt.pcolor(
            combined_map_to_plot,
            cmap=cmap,
            norm=norm,
            edgecolors='k',
            linewidth=3,
        )
        
        # ticks and labels
        ax = plt.gca()
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.xaxis.tick_top()        
        plt.xticks([i+0.5 for i in range(y)], [i+1 for i in range(y)])
        plt.yticks([i+0.5 for i in range(x)], [i+1 for i in range(x)])
        
        # labels
        xmin, ymin = min(combined_map_to_plot.columns), min(combined_map_to_plot.index)
        for i in range(self.label_map.shape[0]):
            for j in range(self.label_map.shape[1]):
                label = self.label_map.iloc[i, j]
                if pd.notna(label):
                    plt.text(j-xmin+0.5, i-ymin+0.5, label, ha='center', va='center', color='black', size=label_size)

        # Legend
        labels = [x.title().replace('Master', 'Master Monster') for x in COLOUR_DF.object.to_list() if x != 'JOIN' and x != 'NOTHING']
        colours = [cmap(COLOUR_DF.id.to_list().index(x)) for x in COLOUR_DF.id.to_list()[1:] if x != 0.5 and x != 99]

        # Create square-shaped handles for the legend
        square_handles = [Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=23, label=label, linewidth=0) 
                          for color, label in zip(colours, labels)]

        plt.legend(
            title='LEGEND',
            title_fontsize=28,
            handles=square_handles,
            handleheight=2,
            loc='upper right',
            bbox_to_anchor=(-0.05, 1.03),
            fontsize=label_size,
            frameon=False
        )

        # ai textblock
        list_of_monsters_to_show = None if not show_up_to_area_n else sorted(list(
            {item.replace(' Master', '').replace(' Boss', '') for i in range(1, show_up_to_area_n+1) for item in self.encounters['monsters'].get(i, [])}
        ))
        if show_boss is None:
            show_boss = show_up_to_area_n >= self.N_ENCOUNTERS if show_up_to_area_n else True
        plt.text(y+2, -0.5, self.ai_summary(show_boss, monsters_to_display=list_of_monsters_to_show), fontsize=14, ha='left', va='top', wrap=True)

        plt.show()

    # tweaking the scenario
    
    def regenerate_map(self, keep_monsters=True, keep_treasures=True):
        # map gen repeatedly calls __init__ so we need to back up monsters, terrain, treasure
        if keep_monsters:
            monsters_obj = self.monsters_obj
            ai_overlord = self.ai_overlord
            monster_encounters = self.encounters['monsters']
        if keep_treasures:
            treasures_obj = self.treasures_obj
            treasure_encounters = self.encounters['treasures']
        
        print('creating map...')
        self.create_map()  # this initialises all layers and attributes

        if keep_monsters:
            self.monsters_obj = monsters_obj
            self.encounters['monsters'] = monster_encounters
            self.ai_overlord = ai_overlord
            self.monster_tiles_obj = Tiles(self.monsters_obj)
        if keep_treasures:
            self.treasures_obj = treasures_obj
            self.encounters['treasures'] = treasure_encounters
        
        self.label_dict = self.ai_overlord.label_dict
        print('distributing monsters...')
        self.distribute_monsters()
        print('distributing terrain...')
        self.create_terrain()
        self.distribute_terrain()
        print('distributing treasures...')
        self.distribute_treasures()
        print('complete!')
        
    def regenerate_monsters(self, keep_monsters=False, keep_archetypes=False):
        if not keep_monsters:
            self.monsters_obj = Monsters(self._n_monsters, self.N_ENCOUNTERS, self._boss, self._sorted_battles, self._use_all_minis)
            self.encounters['monsters'] = {i+1:x for i, x in enumerate(self.monsters_obj.encounters.values())}

        if not keep_archetypes or not keep_monsters:  # can't keep AI if new monsters are chosen (with current funcionality)
            self.ai_overlord = AIOverlord(self.monsters_obj.quest_monsters, self.monsters_obj.quest_boss)
            self.label_dict = self.ai_overlord.label_dict
            
        self.label_map = self.MAP_TEMPLATE.copy()
        self.monster_map = self.MAP_TEMPLATE.copy()
        self.monster_tiles_obj = Tiles(self.monsters_obj)
        print('distributing monsters...')
        self.distribute_monsters()
    
    def regenerate_terrain(self, same_terrain=False, terrain_type=None, use_all=False):
        self.terrain_map = self.MAP_TEMPLATE.copy()
        if not same_terrain:
            self.terrain_tiles_obj = Tiles(TERRAIN_DF)
            self.create_dungeon_entrance(only_terrain=True)
            self.encounters['terrain'] = {k+1:list() for k in range(self.N_ENCOUNTERS)}
            self.create_terrain(terrain_type, use_all)
        print('distributing terrain...')
        self.distribute_terrain()
        
    def regenerate_treasure(self, keep_treasure=False):
        self.treasure_map = self.MAP_TEMPLATE.copy()
        if not keep_treasure:
            self.encounters['treasures'] = {i+1:x for i, x in enumerate(self.treasures_obj.encounters.values())}
            self.treasures_obj = Treasures(self.N_ENCOUNTERS, self._n_treasure_per_encounter, self._always_chest)
        print('distributing treasures...')        
        self.distribute_treasures()


class MonsterAI:
    # TODO: for now master and minions have the same archetype, it is possible to tweak this

    def __init__(self, monster_name, archetype=None, boss=False):
        if monster_name not in list(MONSTER_DF.name):
            raise ThisIsNotAMonsterYouCanChoose
        self.monster_name = monster_name
        self.boss = boss
        self.boss_powers = self.boss_stuff()
        self._archetype_data = self.get_archetype_data(archetype)

    def get_archetype_data(self, archetype):
        columns = ['archetype', 'archetype_first', 'target', 'range_modifier', 'move_modifier', 'special', 'boss_special', 'lt_range', 'at_range', 'gt_range', 'gt_move_range']
        all_eligible_archetypes = ARCHETYPES.loc[ARCHETYPES[self.monster_name].notna(), columns]
        if archetype:
            if archetype not in list(all_eligible_archetypes['archetype']):
                raise ThisIsNotAReasonableChoiceForThatMonsterError
            return all_eligible_archetypes[all_eligible_archetypes['archetype']==archetype]
        return all_eligible_archetypes.sample()

    @property
    def archetype(self):
        return self._archetype_data['archetype'].iloc[0]

    @property
    def target(self):
        return self._archetype_data['target'].iloc[0]
    
    @property
    def range(self):
        range = MONSTER_DF.loc[MONSTER_DF['name']==self.monster_name, 'range'].iloc[0]
        if self._archetype_data['range_modifier'].notnull().iloc[0]:
            if self.boss:
                range += eval(self._archetype_data['range_modifier'].iloc[0])[1]
            else:
                range += eval(self._archetype_data['range_modifier'].iloc[0])[0]
        return range

    @property
    def movement(self):
        movement = MONSTER_DF.loc[MONSTER_DF['name']==self.monster_name, 'movement'].iloc[0]
        if self._archetype_data['move_modifier'].notnull().iloc[0]:
            if self.boss:
                movement += eval(self._archetype_data['move_modifier'].iloc[0])[1]
                if 'move' in self.boss_powers.lower() and '+' in self.boss_powers:
                    movement += eval(self.boss_powers.lower().split('move')[0].strip().split('+')[-1])
            else:
                movement += eval(self._archetype_data['move_modifier'].iloc[0])[0]
        return movement
    
    @property
    def lt_range(self):
        return self._archetype_data['lt_range'].iloc[0]
    
    @property
    def at_range(self):
        return self._archetype_data['at_range'].iloc[0]
    
    @property
    def gt_range(self):
        return self._archetype_data['gt_range'].iloc[0]
    
    @property
    def gt_move_range(self):
        return self._archetype_data['gt_move_range'].iloc[0]

    @property
    def special(self):
        if self.boss:
            return self._archetype_data['boss_special'].iloc[0]
        return self._archetype_data['special'].iloc[0]
    
    @property
    def archetype_first(self):
        return self._archetype_data['archetype_first'].iloc[0]

    def boss_stuff(self):
        possible_boss_stuff = ['4x HP', '2x HP, +3 Armor', '2x HP, +5 Power Dice for Attacks', '2x HP, Undying 2', '2x HP, +3 Move, Dodge',
                               '3x HP, Leech', '2x HP, +3 Range, Aim', '3x HP, Always Battle', 'Dodge, Permanent Stealth, Drain, Leech, +3 Armor']
        return random.choice(possible_boss_stuff)

    def summary(self, label_idx=None):
        text = '------------------------------------------------------------------'
        
        # monster title and modifier text
        label = label_idx+': ' if label_idx else ''
        if self.archetype == 'Normal':
            monster = self.monster_name
        elif self.archetype_first:
            monster = f"{self.archetype} {self.monster_name}"
        else:
            monster = f"{self.monster_name} {self.archetype}"
        text += f'\n{label}{monster}\n------------------------------------------------------------------\n'

        if pd.notnull(self.special):
            text += f'\nSpecial Powers: {self.special}'.title().replace('Hp', 'HP')
        if self.boss:
            text += f'\nBoss Special Powers: {self.boss_powers}'

        text += f'\nTarget: {self.target}'.title().replace('Hp', 'HP')

        attack_type = MONSTER_DF.loc[MONSTER_DF['name']==self.monster_name, 'attack_type'].iloc[0]
        if attack_type == 'melee':
            attack_range = 1 if 'reach' in MONSTER_DF.loc[MONSTER_DF.name==self.monster_name, 'abilities'].iloc[0].lower() else 0
            text += f'\nAttack Range: {attack_range}\n'
        else:
            attack_range = self.range
            text += f'\nAttack Range: {attack_range-1} to {attack_range+1} \n'

        text += '\nCombat AI Instructions\n----------------------'

        if attack_type == 'melee':
            if pd.notnull(self.at_range):
                text += f'\nIf at attack range: {self.at_range}'
            if pd.notnull(self.gt_range):
                text += f'\nIf within {self.movement + attack_range} spaces: {self.gt_range}'
            if pd.notnull(self.gt_move_range):
                text += f'\nOtherwise: {self.gt_move_range}'
        else:
            if pd.notnull(self.lt_range):
                text += f'\nIf closer than {self.range-1} spaces: {self.lt_range}'
            if pd.notnull(self.at_range):
                text += f'\nIf {attack_range-1} to {attack_range+1} spaces: {self.at_range}'
            if pd.notnull(self.gt_range):
                text += f'\nIf within {self.movement + attack_range} spaces: {self.gt_range}'
            if pd.notnull(self.gt_move_range):
                text += f'\nOtherwise: {self.gt_move_range}'
        
        return text + '\n'


class AIOverlord:
    """
    if monster_list_or_dict is a dictionary, 
        'boss' keyword denotes the boss monstertype
        'boss_type_minions' will default to False if not found in the dict, but can be used to also use this type for minions
        this monstertype must be listed with an archetype along with all other monster types
    if monster_list_or_dict is a list,
        just list the monsters you want, and randomness will choose archetypes
        if you want a boss, use boss_monster_name, otherwise there will be no boss
    """
        
    def __init__(self, monsters_list_or_dict, boss_monster_name=None, all_undead=True, archetype=None):
        self.monsters = dict()
        self._archetype = archetype  # TODO: could upgrade this to use list/dict to choose each archetype for each monster manually
        self._not_undead = False
        if type(monsters_list_or_dict) == list:
            for monster in monsters_list_or_dict:
                for _ in range(10):
                    self.monsters[monster] = MonsterAI(monster, self._archetype)
                    if self._not_undead and self.monsters[monster].archetype == 'Undead':
                        continue
                    if all_undead and self.monsters[monster].archetype == 'Undead':
                        self._archetype = 'Undead'
                        break
                    if all_undead and self.monsters[monster].archetype != 'Undead':
                        self._not_undead = True
                    break
            self.boss = None if not boss_monster_name else MonsterAI(boss_monster_name, self._archetype, boss=True)
        elif type(monsters_list_or_dict) == dict:
            if 'boss' in monsters_list_or_dict:
                boss_monster_name = monsters_list_or_dict.pop('boss')
                if 'boss_type_minions' in monsters_list_or_dict:
                    boss_type_minions = monsters_list_or_dict.pop('boss_type_minions')
                else:
                    boss_type_minions = False
                if boss_type_minions:
                    boss_archetype = monsters_list_or_dict['boss']
                else:
                    boss_archetype = monsters_list_or_dict.pop(boss_monster_name)
                self.boss = MonsterAI(boss_monster_name, boss_archetype, boss=True)
            else:
                self.boss = None

            for monster, archetype in monsters_list_or_dict.items():
                self.monsters[monster] = MonsterAI(monster, archetype)
        else:
            raise ThisObjectNeedsBetterMonsterDataError

        self.label_dict = {m:chr(i+65) for i,m in enumerate(list(self.monsters.keys()) + [self.boss.monster_name + ' Boss'])}

    def summary(self, show_boss=True, monsters_to_display=None, print_summary=False):
        # filter results to display
        if monsters_to_display:
            monster_list = monsters_to_display
            if self.boss.monster_name in monster_list:
                show_boss = True if show_boss else False
                if self.boss.monster_name not in self.monsters.keys():
                    monster_list.remove(self.boss.monster_name)
            else:
                show_boss = False
            monster_obj_list = [x for x in self.monsters.values() if x.monster_name in monster_list]
        else:
            monster_obj_list = list(self.monsters.values())

        text = '***************************  MONSTERS  ***************************\n'
        for monster in monster_obj_list:
            text += '\n' + monster.summary(self.label_dict[monster.monster_name])
        if show_boss and self.boss:
            text += '\n*****************************  BOSS  *****************************\n'
            text += '\n' + self.boss.summary(self.label_dict[self.boss.monster_name+' Boss'])
        text += '\n******************************************************************'
        if print_summary:
            print(text)
        else:
            return text

    def boss_summary(self, print_summary=False):
        text = '\n*****************************  BOSS  *****************************\n'
        if self.boss:
            text += '\n' + self.boss.summary(self.label_dict[self.boss.monster_name+' Boss'])
            text += '\n******************************************************************'
        else:
            text += '\nThere is no boss for this quest.'
        if print_summary:
            print(text)
        else:
            return text
    
    @property
    def ai_overlord_attributes(self):
        monsters_dict = dict()
        for monster, obj in self.monsters.items():
            monsters_dict[monster] = obj.archetype
        if self.boss:
            monsters_dict[self.boss.monster_name] = self.boss.archetype
            monsters_dict['boss'] = self.boss.monster_name
            # there is no way to determine 'boss_type_minions' without quest knowledge, so left null/False
        return monsters_dict


class AIQuestMonsters:
    def __init__(self, n_monsters=5, n_encounters=4, boss=True, sorted_battles=True, use_all_minis=False, all_undead=True, archetype=None):
        self.monster_obj = Monsters(n_monsters, n_encounters, boss, sorted_battles, use_all_minis)

    @property
    def ai_overlord_attributes(self):
        monsters_list = list()
        for monster in self.monster_obj.quest_monsters:
            monsters_list.append(monster)
        if self.monster_obj.quest_boss:
            boss = self.monster_obj.quest_boss
        return monsters_list, boss


