- Add more client configurations to the rule-mining dataset.
  
  *- Perform fresh FL runs with slow clients, uneven data distributions, communication delays, etc.*

- Flesh out and evaluate the complete decide-or-defer pipeline.
  
  *- Add a student-policy branch to AP4Fed and a rule dispatcher to the distillation package, with fallback to the LLM.*

- Keep the compact feature set as the baseline, then add configuration context. 
   
  *- Add configuration and workload features when multiple setups are introduced, then check whether they improve rule quality.*

- Compare different workloads. 
  
  *- Test whether rules transfer between FashionMNIST and AG News, noting that the model also changes.*

- Record the client-selector threshold?
  
  *- The archive only contains binary ON/OFF values, while the planned framework distinguishes OFF and 3 different ON thresholds.*