from backend import recommend
from utils import resolve_lines


def main():
    test_inputs = [
        "Invincible (2021)",
        "Dr. Brain (2021)",
        "The Expanse (2015)",
        "Atlanta (2016)",
        "Better Call Saul (2015)",
        "The Resort (2022)",
        "The Peripheral (2022)",
        "Black Bird (2022)",
        "Interview with the Vampire (2022)",
        "Living with Yourself (2019)",
        "We Own This City (2022)",
        "Corporate (2018)",
        "Mob Psycho 100 (2016)",
        "Dead Set (2008)",
        "Humans (2015)",
        "Chernobyl (2019)",
        "Attack the Block (2011)",
        "The Lobster (2015)",
        "The Endless (2017)",
        "The Road (2009)",
        "Super (2010)",
        "Chronicle (2012)",
        "Landscape with Invisible Hand (2023)",
        "Arrival (2016)",
        "Dale and Tucker Vs. Evil (2018)",
        "The Martian (2015)",
        "The World’s End (2013)",
        "The Brand New Testament (2015)",
        "Looper (2012)",
        "Edge of Tomorrow (2014)",
        "Watchmen (2019)",
        "The Shape of Water (2017)",
        "Kaleidoscope (2023)",
        "Stranger Things (2016)",
        "The Handmaid’s Tale (2017)",
        "Des (2020)",
    ]

    likes_txt = "\n".join(test_inputs)
    dislikes_txt = ""
    likes_raw = [t.strip() for t in likes_txt.splitlines() if t.strip()]
    dislikes_raw = [t.strip() for t in dislikes_txt.splitlines() if t.strip()]

    ok_like, miss_like = resolve_lines(likes_raw)
    ok_dis, miss_dis = resolve_lines(dislikes_raw)
    liked_ids = [imdb for _raw, imdb, _ in ok_like]
    disliked_ids = [imdb for _raw, imdb, _ in ok_dis]

    recs = recommend(
        liked_ids=liked_ids,
        disliked_ids=disliked_ids,
    )
    print("Recommendations:")
    for rec in recs:
        #                 "title": m.get("title", "N/A"),
        #                 "year": m.get("year", ""),
        #                 "rating": m.get("rating", ""),
        #                 "norm_rating": m.get("norm_rating", ""),
        #                 "score": round(scores[mid], 4),
        #                 "url": f"https://www.imdb.com/title/{mid}/",
        print(
            f"{rec['title']} ({rec['year']}) - {rec['rating']} - {rec['norm_rating']} - {rec['score']}"
        )

if __name__ == "__main__":
    main()