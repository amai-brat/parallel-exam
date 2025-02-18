## Решение двумерного нелинейного уравнения теплопроводности методом продольно-поперечной прогонки


```bash
# последовательно
g++ ./seq2.cpp -o seq2 -Wall
./seq2 > seq.csv

# параллельно (MPI)
mpic++ -Wall ./parallel2.1.cpp -o ./parallel2.1 
mpiexec -n 30 --map-by :OVERSUBSCRIBE ./parallel2.1 > parallel.csv
```

sns.heatmap из полученного csv:
![heatmap](https://i.imgur.com/QtvzrLV.png)


Вдохновлялся [статьёй](https://habr.com/ru/articles/707462/)
