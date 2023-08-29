import matplotlib.pyplot as plt
from pyquaternion import Quaternion


class BaseRender:
  """
    BaseRender class
    """

  def __init__(self, figsize=(10, 10)):
    self.figsize = figsize
    self.fig, self.axes = None, None

  def reset_canvas(self, dx=1, dy=1, tight_layout=False):
    # close current plot.
    plt.close()
    # set current axis off.
    plt.gca().set_axis_off()
    plt.axis('off')
    # Create a new figure and axes with the specified dimensions and figsize
    # row: 2 col: 3
    self.fig, self.axes = plt.subplots(dx, dy, figsize=self.figsize)
    if tight_layout:
      plt.tight_layout()
  
  def get_axes_size(self):
    # 获取与 axes 关联的 Figure 对象的大小（单位：英寸）
    fig_width_in_inches, fig_height_in_inches = self.fig.get_size_inches()
    
    # 获取 Figure 对象的 DPI 值
    dpi = self.fig.dpi
    
    # 将英寸转换为像素
    fig_width_in_pixels = int(fig_width_in_inches * dpi)
    fig_height_in_pixels = int(fig_height_in_inches * dpi)
    
    return fig_width_in_pixels, fig_height_in_pixels

  def close_canvas(self):
    plt.close()

  def save_fig(self, filename):
    # Adjust subplot parameters to remove margins
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(filename)
