export function Icon({name,size=20}:{name:string;size?:number}) {
  const paths:Record<string,React.ReactNode>={
    upload:<><path d="M12 16V3m-5 5 5-5 5 5"/><path d="M4 15v5h16v-5"/></>,
    file:<><path d="M14 2H5v20h14V7Z"/><path d="M14 2v6h5M8 13h8M8 17h6"/></>,
    search:<><circle cx="10" cy="10" r="6"/><path d="m15 15 6 6"/></>,
    play:<path d="m8 4 13 8-13 8Z"/>,pause:<><path d="M8 4v16M16 4v16"/></>,
    download:<><path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5"/></>,
    check:<path d="m5 12 4 4L19 6"/>,close:<path d="m6 6 12 12M18 6 6 18"/>,
    clock:<><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
    left:<path d="m15 5-7 7 7 7"/>,right:<path d="m9 5 7 7-7 7"/>,
    trash:<><path d="M3 6h18M9 6V3h6v3M6 6l1 15h10l1-15M10 10v7M14 10v7"/></>,
    sound:<><path d="M4 9v6h4l5 4V5L8 9ZM17 8q6 4 0 8"/></>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]||paths.file}</svg>;
}
