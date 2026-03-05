
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class Detectie(Base):
    __tablename__ = 'detectii'
    
    # 1. ID și Timp
    id = Column(Integer, primary_key=True, autoincrement=True)
    data_ora = Column(DateTime, default=datetime.now)
    
    # 2. Informații Generale
    tip_obiect = Column(String(50), nullable=False)
    
    # 3. Detalii Vehicule
    numar_inmatriculare = Column(String(20), nullable=True)
    culoare = Column(String(30), nullable=True)
    
    # 4. Detalii Animale
    rasa_animal = Column(String(50), nullable=True)
    incredere_rasa = Column(Float, nullable=True)
    
    # 5. Detalii Persoane
    gen_persoana = Column(String(20), nullable=True)
    incredere_gen = Column(Float, nullable=True)

    def to_dict(self):
        """Pentru afișare statistici"""
        return {
            'id': self.id,
            'data': self.data_ora.strftime('%Y-%m-%d %H:%M:%S'),
            'tip': self.tip_obiect,
            'nr_inmatriculare': self.numar_inmatriculare,
            'culoare': self.culoare,
            'rasa': f"{self.rasa_animal} ({self.incredere_rasa:.0%})" if self.rasa_animal else None,
            'gen': f"{self.gen_persoana} ({self.incredere_gen:.0%})" if self.gen_persoana else None
        }

# Configurare Bază de Date
engine = create_engine('sqlite:///detectii.db', echo=False, pool_pre_ping=True)
Session = sessionmaker(bind=engine)

def init_db():
    """Creează tabelul dacă nu există."""
    try:
        Base.metadata.create_all(engine)
        print("[DATABASE] Baza de date simplificată inițializată.")
        return True
    except Exception as e:
        print(f"[DATABASE] Eroare init: {e}")
        return False

def salveaza_detectie(data):
    """
    Salvează doar câmpurile esențiale.
    Ignoră coordonatele sau alte date trimise din main.py.
    """
    session = Session()
    try:
        detectie = Detectie(
            tip_obiect=data.get('tip_obiect'),
            
            # Vehicule
            numar_inmatriculare=data.get('numar_inmatriculare'),
            culoare=data.get('culoare'),
            
            # Animale
            rasa_animal=data.get('rasa_animal'),
            incredere_rasa=data.get('incredere_rasa'),
            
            # Persoane
            gen_persoana=data.get('gen_persoana'),
            incredere_gen=data.get('incredere_gen')
        )
        
        session.add(detectie)
        session.commit()
        
        # Log scurt în consolă
        info_extra = ""
        if detectie.numar_inmatriculare: info_extra = f"| {detectie.numar_inmatriculare}"
        elif detectie.rasa_animal: info_extra = f"| {detectie.rasa_animal}"
        elif detectie.gen_persoana: info_extra = f"| {detectie.gen_persoana}"
        
        print(f"[DB] Salvat: {detectie.tip_obiect} {info_extra}")
        return detectie.id
        
    except Exception as e:
        print(f"[DB] Eroare: {e}")
        return None
    finally:
        session.close()

def get_statistici():
    """Returnează ultimele înregistrări pentru afișare la final."""
    session = Session()
    try:
        rec = session.query(Detectie).order_by(Detectie.id.desc()).limit(10).all()
        total = session.query(Detectie).count()
        return {'ultimele': [r.to_dict() for r in rec], 'total': total}
    except:
        return {}
    finally:
        session.close()

if __name__ == "__main__":
    init_db()