document$.subscribe(() => {
  // Re-initialise Mermaid on every navigation (needed with navigation.instant).
  mermaid.initialize({ startOnLoad: false })
  mermaid.run({ querySelector: ".mermaid" })
})
