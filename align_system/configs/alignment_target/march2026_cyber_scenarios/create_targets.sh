#!/bin/bash

for attr in integrity availability sensitivity_to_doubt protocol organization_reputation intelligence; do
  for value in 0.0 1.0; do
    filename="${attr}-${value}.yaml"
    cat > "$filename" << EOF
id: ADEPT-March2026-${attr}-${value}
kdma_values:
- kdes: null
  kdma: ${attr}
  value: ${value}
EOF
  done
done
