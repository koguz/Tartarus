for /l %%N in (1, 1, 8) do (
	for /l %%S in (4, 1, 6) do (
		for /l %%P in (1, 1, 2) do (
			for /l %%X in (1, 1, 5) do (
				.\Tartarus00.exe %%N %%S %%P 0 %%X
				.\Tartarus00.exe %%N %%S %%P 1 %%X
				.\Tartarus00.exe %%N %%S %%P 2 %%X
			)
		)
	)
)