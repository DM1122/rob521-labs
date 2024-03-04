# Lab 2 Task Flowchart
```mermaid
flowchart TB
    start((Start))
    task1[Task 1 David]
    task2[Task 2 David]
    task3[Task 3 Daye]
    task4[Task 4 Daye]
    task5[Task 5 Davin]
    task6[Task 6 David]
    task7[Task 7 Davin]
    task8[Task 8 Daye]
    endd((End))
    
    start --> task1 & task2
    
    task1 --> task3
    task2 --> task3

    task1 --> task8
    task2 --> task8

    task3 --> task4 & task5
    task4 --> task6
    task5 --> task6
    task6 --> task7 --> endd
```