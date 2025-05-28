

# Grafichina per la progress bar
BAR_WIDTH = 30
BAR_BORDERS= ('[', ']')
BAR_CHAR= '='

class ProgressBar:
  def __init__(self, max_width: int= BAR_WIDTH, char_borders: tuple[str, str]= BAR_BORDERS, progress_char: str= BAR_CHAR):

    self.max_width = max_width
    self.char_borders = char_borders
    self.progress_char = progress_char

  def make_progress(self, curr_prog: int, max_width: int) -> str:

    progress_ratio= curr_prog /  max_width
    return f"{self.char_borders[0]}{self.progress_char * int(self.max_width * progress_ratio):<{BAR_WIDTH}}{self.char_borders[1]}"