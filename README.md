# annanalysis
This is a repo containing work on my thesis concerning approximate nearest neighbors search techniques (ANN, ANNS)

## What's the approximate nearest neighbour search?

Imagine you have to search a hudge music database to find a songs most similar to song that you have just listened to.
It would require too much time and resources to check every single song of the database for similarity and you are a bit impatient!
That's where aprroximate search pops in. 
The idea is not to find exact nearest neighbour (although it might really find one), but to find those that are "close enough".
Therefore the database is usually  split into chunks, and real search happens in chunks that are likley to contain songs you want hear.
Making and searching those chunks - that is what AAN techniques are about!

## Where can I find more about ANN algortihms?
This guy is gathering a nice protion of ANN techniques and making comparison between them:https://github.com/erikbern/ann-benchmarks
These people have a cool library and implementations of techniques, checkout HNSW idea: https://github.com/nmslib/nmslib



