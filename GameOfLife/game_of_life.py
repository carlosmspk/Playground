import time
import os


class Cell:
    def __init__(self, alive : bool = False):
        self.alive = alive

    def get_next_state(self, number_of_living_cells) -> bool:
        if self.alive:
            if number_of_living_cells < 2 or number_of_living_cells > 3:
                return False
            return True
        else:
            if number_of_living_cells == 3:
                return True
            return False
    
    def is_alive(self) -> bool:
        return self.alive

    def die(self):
        self.alive = False
    
    def live(self):
        self.alive = True
     

class GameOfLife:
    def __init__(self, grid_length=10, grid_width=10, initially_alive_cells : set = {(0,0)}) -> None:
        self.alive_cells_exist = True
        self.population = []
        self.next_states = []
        self.iterations = 0
        self.grid_length = grid_length
        self.grid_width = grid_width
        self.reset(initially_alive_cells)
        self.NEIGHBORHOOD_OFFSETS = (
            (-1,-1),(-1,0),(-1,1),
            (0,-1) ,       (0,1),
            (1,-1) ,(1,0), (1,1))

    def reset(self, alive_positions : set):
        for row in range (self.grid_length):
            population_row = []
            next_state_row = []
            for col in range (self.grid_width):
                if (row, col) in alive_positions:
                    population_row.append(Cell(alive=True))
                else:
                    population_row.append(Cell())
                next_state_row.append(None)
            self.population.append(population_row)
            self.next_states.append(next_state_row)
        self.alive_cells_exist = True


    def run_generation(self):
        self.iterations += 1
        for row, cell_row in enumerate(self.population):
            for col, cell in enumerate(cell_row):
                position = (row, col)
                self.next_states[position[0]][position[1]] = cell.get_next_state(self.get_alive_cells_in_neighborhood(position))
        self.alive_cells_exist = False
        for row, cell_row in enumerate(self.population):
            for col, cell in enumerate(cell_row):
                position = (row, col)
                if self.next_states[position[0]][position[1]]:
                    cell.live()
                    self.alive_cells_exist = True
                else:
                    cell.die()


    def get_alive_cells_in_neighborhood(self, position : tuple) -> int:
        n = 0
        for pos_offset in self.NEIGHBORHOOD_OFFSETS:
            row = position[0] + pos_offset[0]
            col = position[1] + pos_offset[1]
            if row < 0 or row > self.grid_length-1:
                continue
            elif col < 0 or col > self.grid_width-1:
                continue
            elif self.population[row][col].is_alive():
                n += 1
        return n

    def print_board(self):
        print('\n\n\n\n\n\n')
        print('_'*(self.grid_width+2))
        for cell_row in self.population:
            row_to_print = '|'
            for cell in cell_row:
                assert isinstance(cell, Cell)
                if cell.is_alive():
                    row_to_print += 'O'
                else:
                    row_to_print += ' '
            row_to_print += '|'
            print(row_to_print)
        print('‾'*(self.grid_width+2))

    def run (self, timestep : float = 1.0):
        now = time.time()
        self.print_board()
        time.sleep(max(0, timestep - (time.time() - now)))
        while self.alive_cells_exist:
            now = time.time()
            self.run_generation()
            self.print_board()
            time.sleep(max(0, timestep - (time.time() - now)))
            os.system("cls")
        print(f"Cells lived for {self.iterations} generations.")


class Patterns:
    L = {(7,5),(7,6),(7,7),(5,5),(6,5)}
    glider = {(0,1), (1,2), (2,0), (2,1), (2,2)}
    r_pentomino = {(0,1),(0,2),(1,0),(1,1),(2,1)}
    gosper_gun = {
        (4,0), (4,1), (5,0), (5,1), #left square
        (2,34), (2,35), (3,34), (3,35), #right square
        (2,12), (2,13), (3,11), (3,15), (4,10), (4,16), (5,10), (5,14), (5,16), (5,17), (6,10), (6,16), (7,11), (7,15), (8,12), (8,13), # weird bubble
        (0,24), (1,22), (1,24), (2,20), (2,21), (3,20), (3,21), (4,20), (4,21), (5,22), (5,24), (6,24) #weird space ship
    }

    @staticmethod
    def offset(pattern, offset) -> set:
        new_set = set()
        for position in pattern:
            new_set.add((position[0]+offset[0],position[1]+offset[1]))
        return new_set

    @staticmethod
    def flip(pattern, orientation = 'Diagonal') -> set:
        new_set = set()
        if orientation == 'Diagonal':
            for position in pattern:
                new_set.add((position[1],position[0]))
        elif orientation == 'Horizontal':
            max_row = None
            min_row = None
            for row, _ in pattern:
                if max_row is None or max_row < row:
                    max_row = row
                if min_row is None or min_row > row:
                    min_row = row
            gap = max_row - min_row
            if gap % 2:
                flip_row = gap/2 + min_row
            else:
                flip_row = int(gap/2) + min_row

            for row, col in pattern:
                # e.g. row=2, flip_row=1 -> reflected_row=0
                reflected_row = row - (row - flip_row)*2
                new_set.add((reflected_row, col))
                
            
        elif orientation == 'Vertical':
            max_col = None
            min_col = None
            for _, col in pattern:
                if max_col is None or max_col < col:
                    max_col = col
                if min_col is None or min_col > col:
                    min_col = col
            gap = max_col - min_col
            if gap % 2:
                flip_col = gap/2 + min_col
            else:
                flip_col = int(gap/2) + min_col

            for row, col in pattern:
                # e.g. row=2, flip_row=1 -> reflected_row=0
                reflected_col = col - (col - flip_col)*2
                new_set.add((row, reflected_col))
                
        
        return new_set

    @staticmethod
    def print(pattern) -> None:
        min_row = None
        max_row = None
        max_col = None
        min_col = None
        for row, col in pattern:
            if max_col is None or max_col < col:
                max_col = col
            if min_col is None or min_col > col:
                min_col = col
            if max_row is None or max_row < row:
                max_row = row
            if min_row is None or min_row > row:
                min_row = row
        
        if min_row is None or max_row is None or min_col is None or max_col is None:
            return
                
        for row in range(max_row-min_row+1):
            row_string = ''
            for col in range (max_col-min_col+1):
                if (row+min_row, col+min_col) in pattern:
                    row_string += 'O'
                else:
                    row_string += ' '
            print (row_string)




if __name__ == '__main__':
    initial_pattern = Patterns.gosper_gun
    
    gol = GameOfLife(30,100, initial_pattern)
    gol.run(timestep=0.1)

    