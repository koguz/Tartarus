for /l %%N in (1, 1, 8) do (
	for /l %%S in (4, 1, 16) do (
		for /l %%L in (0, 1, 9) do .\Tartarus00.exe %%N %%S %%L 
	)
)