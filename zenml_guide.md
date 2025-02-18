
# ZenML Stack Management Guide

This guide provides quick commands to manage stacks and artifact stores in ZenML. Follow these steps to copy, register, delete, and set your stacks and artifact stores efficiently.

---

## Stack Management

### 1. Copy a Stack
To copy the default stack to a new stack named `my_stack`:
```bash
zenml stack copy default my_stack
```

### 2. Update a Stack with a New Artifact Store
To update a stack `my_stack` with a new artifact store:
```bash
zenml stack update my_stack -a local_artifact_store
```

### 3. Set a New Default Stack
To set `my_stack` as the default stack:
```bash
zenml stack set my_stack
```

### 4. Delete a Stack
To delete an existing stack:
```bash
zenml stack delete <stack_name>
```

---

## Artifact Store Management

### 1. List All Artifact Stores
To list all registered artifact stores:
```bash
zenml artifact-store list
```

### 2. Register a New Artifact Store
To register a local artifact store with a specific path:
```bash
zenml artifact-store register local_artifact_store --flavor=local --path="C:\Users\ChiragPassi\Downloads\Projects\ML-Ops\ML Run\zenml_artifacts"
```

### 3. Delete an Artifact Store
To delete an artifact store named `new_local_artifact_store`:
```bash
zenml artifact-store delete new_local_artifact_store
```

---

## Example Usage

1. **Copy the Default Stack**:  
   Create a new stack by copying the default configuration:
   ```bash
   zenml stack copy default my_stack
   ```

2. **Register a New Artifact Store**:  
   Register a local artifact store to use for your stack:
   ```bash
   zenml artifact-store register local_artifact_store --flavor=local --path="C:\path_to_artifacts"
   ```

3. **Update Your Stack**:  
   Link the newly registered artifact store to the stack:
   ```bash
   zenml stack update my_stack -a local_artifact_store
   ```

4. **Set Your Stack as Default**:  
   Use the stack `my_stack` as your default stack:
   ```bash
   zenml stack set my_stack
   ```

5. **Delete Unused Stacks or Artifact Stores**:  
   Clean up unused stacks or artifact stores as needed.

---

This guide provides essential commands for efficient ZenML stack and artifact store management. Follow the steps to streamline your workflow.
