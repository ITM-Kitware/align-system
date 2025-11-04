#!/bin/bash

for attr in empowerment protocol availability resource_usage prevention confidentiality personal_reputation organization_reputation sensitivity_to_doubt; do
  for value in 0.0 1.0; do
    filename="${attr}-${value}.yaml"
    cat > "$filename" << EOF
id: ADEPT-October2025-${attr}-${value}
kdma_values:
- kdes: null
  kdma: ${attr}
  value: ${value}
EOF
  done
done
