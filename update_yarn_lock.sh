update() {
  rm yarn.lock && yarn
}

for dir in */; do
    printf '%s\n' "$dir"
    cd "$dir"
    update
    cd ".."
done

