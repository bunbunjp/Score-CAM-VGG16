import os

from icrawler.builtin import GoogleImageCrawler
from typing import List, Tuple


class Configure:
    NAMES: List[str] = [
        "高坂穂乃果",
        "南ことり",
        "小泉花陽",
        "園田海未",
        "星空凛",
        "東條希",
        "絢瀬絵里",
        "西木野真姫",
        "矢澤にこ",
        "高海千歌",
        "渡辺曜",
        "黒澤ルビィ",
        "松浦果南",
        "黒澤ダイヤ",
        "国木田花丸",
        "桜内梨子",
        "津島善子",
        "小原鞠莉",
        "上原歩夢",
        "中須かすみ",
        "桜坂しずく",
        "朝香果林",
        "宮下愛",
        "近江彼方",
        "優木せつ菜",
        "エマ・ヴェルデ",
        "天王寺璃奈"
    ]
    IMAGE_SIZE: Tuple[int, int] = (64, 64)
    ORIGIN_IMAGE_LIMIT: int = 70


if __name__ == '__main__':
    # idx: int = 0
    for idx, name in enumerate(Configure.NAMES):
        crawler: GoogleImageCrawler = GoogleImageCrawler(storage={
            'root_dir': os.path.join('images', str(idx))
        })
        crawler.crawl(keyword=name, max_num=50)
