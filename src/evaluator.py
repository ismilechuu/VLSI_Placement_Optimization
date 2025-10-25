from typing import List, Optional
from .chip import Chip, Net

def calculate_HPWL(chip: Chip, net: Optional[Net] = None) -> float:
    """
    HPWL calculator (ฟังก์ชันเดียวใช้ได้ 2 โหมด)
    - calculate_HPWL(chip)            -> รวมทุกเน็ต (Total HPWL)
    - calculate_HPWL(chip, net=...)   -> เฉพาะเน็ตเดียว
    """
    def _hpwl_one(n: Net) -> float:
        """คำนวณ HPWL ของ net เดียว"""
        xs: List[float] = []
        ys: List[float] = []
        
        for name in n.modules:
            mod = chip.modules.get(name)
                
            x, y = mod.get_position()
            xs.append(x)
            ys.append(y)
        
        # ต้องมีอย่างน้อย 2 modules ที่ถูก place
        if len(xs) < 2:
            return 0.0
            
        return (max(xs) - min(xs)) + (max(ys) - min(ys))

    # Single net mode
    if net is not None:
        return _hpwl_one(net)

    # Total HPWL mode
    return sum(_hpwl_one(n) for n in chip.get_all_nets())
