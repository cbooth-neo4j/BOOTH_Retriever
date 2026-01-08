# Manual Test Questions

Selected from `prepared_corpus.json` for front-end testing.

---

## 1. Comparison Question (Simple)

**Question:** Were Scott Derrickson and Ed Wood of the same nationality?

**Expected Answer:** yes

*Tests the agent's ability to look up two entities and compare a property.*

---

## 2. Bridge Question (Multi-hop Reasoning)

**Question:** What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?

**Expected Answer:** Chief of Protocol

*Requires finding the actress first, then retrieving her government role — classic multi-hop reasoning.*

---

## 3. Bridge Question (Entity + Venue)

**Question:** The arena where the Lewiston Maineiacs played their home games can seat how many people?

**Expected Answer:** 3,677 seated

*Tests knowledge graph traversal: team → arena → capacity.*

