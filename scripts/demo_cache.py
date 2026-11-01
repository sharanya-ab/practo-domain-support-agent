import json
from pathlib import Path
from governance.runtime import cached_grounded_answer, CACHE, CACHE_STATS
def run():
    CACHE.clear(); CACHE_STATS.update(calls=0,hits=0)
    q="What is the appointment cancellation window?"
    first=cached_grounded_answer(q); before=dict(CACHE_STATS)
    second=cached_grounded_answer("  WHAT IS THE APPOINTMENT CANCELLATION WINDOW?  "); after=dict(CACHE_STATS)
    result={"before_repeat":before,"after_repeat":after,"same_answer":first==second,
            "evidence":"calls remains 1 while hits increments to 1"}
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/cache_demo.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result
if __name__=="__main__": print(json.dumps(run(),indent=2))
