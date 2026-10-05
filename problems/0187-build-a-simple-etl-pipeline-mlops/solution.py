class ETLpipeline:
    
    def __init__(self, csv_text: str):
        self.csv_text = csv_text
        
    def extract(self) -> list[dict]:
        """Extract data from CSV text."""
        lines = [
            line.strip() for line in self.csv_text.strip().split("\n") if line.strip()  # ignore blank lines
        ]

        header = [h.strip() for h in lines[0].split(",")]
        data = []

        for line in lines[1:]:
            values = [v.strip() for v in line.split(",")]
            if len(values) != len(header):
                continue  # malformed row
            data.append(dict(zip(header, values)))
        return data
    
    def transform(self, data: list[dict]) -> list[dict]:    
        """Transform data by filtering and converting types."""
        purchases = []
        for row in data:
            if row.get("event_type") != "purchase":
                continue
            try:
                value = float(row["value"])
            except (ValueError, TypeErro