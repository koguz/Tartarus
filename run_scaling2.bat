@echo off

REM === TartarusD2 varying P and L ===
for %%P in (64 128) do (
    for %%L in (3 5 7 9) do (
        TartarusD2 4096 16 3000 %%P %%L
    )
)



REM === TartarusTestAll varying P and L ===
for %%P in (64 128) do (
    for %%L in (3 5 7 9) do (
        TartarusTestAll.exe txt/b-D2-4096-16-3000-%%P-%%L.txt 16
    )
)

