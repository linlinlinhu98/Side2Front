from .base import FaceFrontalizer, FrontalizationResult
from .anime_face import AnimeFaceFrontalizer

__all__ = ["FaceFrontalizer", "FrontalizationResult",
           "AnimeFaceFrontalizer"]

try:
    from .real_face import RealFaceFrontalizer
    __all__.append("RealFaceFrontalizer")
    REAL_MODE_AVAILABLE = True
except ImportError:
    REAL_MODE_AVAILABLE = False

try:
    from .gan_frontalizer import GANFrontalizer, create_gan_frontalizer
    __all__.append("GANFrontalizer")
    GAN_MODE_AVAILABLE = True
except ImportError:
    GAN_MODE_AVAILABLE = False
    create_gan_frontalizer = None
